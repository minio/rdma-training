# Copyright 2026 MinIO, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Measurement helpers: proof-of-RDMA, host CPU, GPU, and NIC/RDMA counters.

The most important thing here is :class:`RDMAWitness`. Because libminiocpp
silently falls back to HTTP when RDMA is declined, a benchmark that trusts its
own ``--backend rdma`` flag can end up measuring TCP twice and reporting the
difference as noise. Every RDMA run in this project brackets its work in an
``RDMAWitness`` and refuses to emit a result unless the server's RDMA byte
counters actually moved by roughly the expected amount.
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import time
import urllib.request
from dataclasses import dataclass, field, asdict
from typing import Iterable, Optional

_METRIC_RE = re.compile(r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>[^\s]+)")

RDMA_METRICS_PATH = "/minio/metrics/v3/api/rdma"


def _scrape(url: str, timeout: float = 10.0) -> dict[str, float]:
    """Parse a Prometheus text exposition into {metric_name: summed_value}.

    Label sets are collapsed by summing, which is what we want: these counters
    are per-server and we care about the cluster total.
    """
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    out: dict[str, float] = {}
    for line in body.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _METRIC_RE.match(line)
        if not m:
            continue
        try:
            v = float(m.group("value"))
        except ValueError:
            continue
        out[m.group("name")] = out.get(m.group("name"), 0.0) + v
    return out


@dataclass
class RDMACounters:
    read_bytes: float = 0.0
    read_ops: float = 0.0
    write_bytes: float = 0.0
    write_ops: float = 0.0

    @classmethod
    def from_scrape(cls, d: dict[str, float]) -> "RDMACounters":
        return cls(
            read_bytes=d.get("minio_api_rdma_read_bytes_total", 0.0),
            read_ops=d.get("minio_api_rdma_read_ops_total", 0.0),
            write_bytes=d.get("minio_api_rdma_write_bytes_total", 0.0),
            write_ops=d.get("minio_api_rdma_write_ops_total", 0.0),
        )

    def __sub__(self, other: "RDMACounters") -> "RDMACounters":
        return RDMACounters(
            read_bytes=self.read_bytes - other.read_bytes,
            read_ops=self.read_ops - other.read_ops,
            write_bytes=self.write_bytes - other.write_bytes,
            write_ops=self.write_ops - other.write_ops,
        )


class RDMAWitness:
    """Bracket a workload and attest whether the server actually served it over RDMA.

    Usage::

        with RDMAWitness(["aistor1:9000", "aistor2:9000"]) as w:
            ... do transfers ...
        w.require(read_bytes=expected_bytes)   # raises if RDMA did not carry it

    ``require`` allows a tolerance because the server counts payload bytes while
    the caller usually knows only the logical object size; small headers and
    partial-range reads make an exact match unrealistic.
    """

    def __init__(self, servers: Iterable[str], *, scheme: str = "http") -> None:
        self.urls = [f"{scheme}://{s}{RDMA_METRICS_PATH}" for s in servers]
        self.before: Optional[RDMACounters] = None
        self.after: Optional[RDMACounters] = None
        self.delta: Optional[RDMACounters] = None
        self.unreachable: list[str] = []

    def _sample(self) -> RDMACounters:
        total: dict[str, float] = {}
        self.unreachable = []
        for u in self.urls:
            try:
                for k, v in _scrape(u).items():
                    total[k] = total.get(k, 0.0) + v
            except Exception as exc:  # a node being down is itself a finding
                self.unreachable.append(f"{u}: {exc}")
        return RDMACounters.from_scrape(total)

    def __enter__(self) -> "RDMAWitness":
        self.before = self._sample()
        return self

    def __exit__(self, *_exc) -> None:
        self.after = self._sample()
        self.delta = self.after - self.before

    def require(
        self,
        *,
        read_bytes: float = 0.0,
        write_bytes: float = 0.0,
        tolerance: float = 0.5,
    ) -> None:
        """Raise unless the RDMA counters moved by at least ``tolerance`` of expected.

        A default tolerance of 0.5 is deliberately loose: it is here to catch the
        catastrophic case (RDMA silently disabled, counters flat at zero), not to
        audit byte accounting.
        """
        if self.delta is None:
            raise RuntimeError("RDMAWitness.require() called before the context exited")
        problems = []
        if read_bytes > 0 and self.delta.read_bytes < read_bytes * tolerance:
            problems.append(
                f"expected >= {read_bytes * tolerance:.0f} RDMA read bytes, "
                f"server counted {self.delta.read_bytes:.0f}"
            )
        if write_bytes > 0 and self.delta.write_bytes < write_bytes * tolerance:
            problems.append(
                f"expected >= {write_bytes * tolerance:.0f} RDMA write bytes, "
                f"server counted {self.delta.write_bytes:.0f}"
            )
        if problems:
            raise RuntimeError(
                "RDMA was NOT used for this workload (the client silently fell back "
                "to HTTP, so this result would be meaningless):\n  - "
                + "\n  - ".join(problems)
                + (f"\n  unreachable metric endpoints: {self.unreachable}" if self.unreachable else "")
            )

    def as_dict(self) -> dict:
        return {
            "rdma_read_bytes": self.delta.read_bytes if self.delta else None,
            "rdma_read_ops": self.delta.read_ops if self.delta else None,
            "rdma_write_bytes": self.delta.write_bytes if self.delta else None,
            "rdma_write_ops": self.delta.write_ops if self.delta else None,
            "metrics_unreachable": self.unreachable,
        }


# --------------------------------------------------------------------- host ---


def _read_proc_stat() -> tuple[float, float]:
    """Return (busy_jiffies, total_jiffies) across all CPUs."""
    with open("/proc/stat") as fh:
        parts = fh.readline().split()
    vals = [float(x) for x in parts[1:]]
    idle = vals[3] + (vals[4] if len(vals) > 4 else 0.0)  # idle + iowait
    total = sum(vals)
    return total - idle, total


class CPUSampler:
    """Whole-host CPU utilisation over an interval, expressed in cores.

    "Cores consumed" is the number that matters for the RDMA story: the claim is
    not that RDMA is a bit faster but that it delivers bytes while leaving the
    CPU free for the rest of the training pipeline.
    """

    def __init__(self) -> None:
        self.ncpu = os.cpu_count() or 1
        self._busy0 = self._total0 = 0.0

    def __enter__(self) -> "CPUSampler":
        self._busy0, self._total0 = _read_proc_stat()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_exc) -> None:
        busy1, total1 = _read_proc_stat()
        self.wall = time.perf_counter() - self._t0
        dtotal = total1 - self._total0
        self.busy_fraction = ((busy1 - self._busy0) / dtotal) if dtotal > 0 else 0.0
        self.cores_used = self.busy_fraction * self.ncpu

    def as_dict(self) -> dict:
        return {
            "cpu_cores_total": self.ncpu,
            "cpu_cores_used": round(getattr(self, "cores_used", 0.0), 3),
            "cpu_busy_fraction": round(getattr(self, "busy_fraction", 0.0), 5),
        }


def nic_counters(devs: Iterable[str]) -> dict[str, int]:
    """rx/tx bytes for the given netdevs, from sysfs."""
    out: dict[str, int] = {}
    for d in devs:
        for k in ("rx_bytes", "tx_bytes"):
            p = f"/sys/class/net/{d}/statistics/{k}"
            try:
                with open(p) as fh:
                    out[f"{d}.{k}"] = int(fh.read().strip())
            except OSError:
                pass
    return out


def rdma_hw_counters(dev: str = "mlx5_0", port: int = 1) -> dict[str, int]:
    """RDMA hardware counters. Rising drop/retry counters mean a lossy fabric.

    ``out_of_buffer``, ``packet_seq_err`` and ``local_ack_timeout_err`` climbing
    under load is the signature of a fabric that is not actually lossless, which
    caps RDMA throughput and must be reported alongside any result.
    """
    base = f"/sys/class/infiniband/{dev}/ports/{port}/hw_counters"
    out: dict[str, int] = {}
    try:
        for name in os.listdir(base):
            try:
                with open(os.path.join(base, name)) as fh:
                    out[name] = int(fh.read().strip())
            except (OSError, ValueError):
                pass
    except OSError:
        pass
    return out


def gpu_info() -> list[dict]:
    """Static GPU inventory via nvidia-smi (no pynvml dependency required)."""
    try:
        q = "index,name,memory.total,driver_version"
        r = subprocess.run(
            ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True,
        )
    except Exception:
        return []
    gpus = []
    for line in r.stdout.strip().splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) >= 4:
            gpus.append({"index": int(f[0]), "name": f[1], "memory_mib": int(f[2]), "driver": f[3]})
    return gpus


@dataclass
class RunRecord:
    """One benchmark run, serialised to results/<name>.json."""

    benchmark: str
    backend: str
    params: dict = field(default_factory=dict)
    results: dict = field(default_factory=dict)
    environment: dict = field(default_factory=dict)
    rdma_witness: dict = field(default_factory=dict)
    host: str = field(default_factory=socket.gethostname)
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S%z"))

    def save(self, directory: str) -> str:
        os.makedirs(directory, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = os.path.join(directory, f"{self.benchmark}-{self.backend}-{stamp}.json")
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=2, sort_keys=True)
        return path
