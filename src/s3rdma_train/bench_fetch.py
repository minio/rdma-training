#!/usr/bin/env python3
# Copyright 2026 MinIO, Inc.
# SPDX-License-Identifier: Apache-2.0

"""B1 — raw object GET throughput into GPU memory.

Establishes the ceiling: what each transport can deliver at all, and at what CPU
cost. Everything downstream (loader throughput, end-to-end step time) is bounded
by this, so it runs first.

    python -m s3rdma_train.bench_fetch --backend rdma --sizes 4,16,64,256 --concurrency 1,8,32

For each (size, concurrency) cell we fetch objects into pre-allocated CUDA
tensors and report GB/s, ops/s, latency percentiles, and host CPU cores consumed.
CPU cost is not a footnote: RDMA's claim is not merely "faster" but "delivers
bytes while leaving the CPU free for the rest of the training pipeline", and that
only shows up if you measure it.

Every RDMA cell is bracketed by RDMAWitness and **fails** if the server's RDMA
byte counters did not move, because libminiocpp falls back to HTTP silently.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch

from .metrics import CPUSampler, RDMAWitness, RunRecord, gpu_info, nic_counters, rdma_hw_counters
from .s3 import StoreConfig, make_store


def _percentiles(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {}
    s = sorted(xs)

    def q(p: float) -> float:
        if len(s) == 1:
            return s[0]
        i = min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))
        return s[i]

    return {
        "min_ms": s[0] * 1e3,
        "p50_ms": q(0.50) * 1e3,
        "p90_ms": q(0.90) * 1e3,
        "p99_ms": q(0.99) * 1e3,
        "max_ms": s[-1] * 1e3,
        "mean_ms": statistics.fmean(s) * 1e3,
    }


def prepare_objects(store, prefix: str, size_bytes: int, count: int, device: str) -> list[str]:
    """Ensure ``count`` objects of exactly ``size_bytes`` exist; return their keys.

    Written once and reused across backends so both transports read byte-identical
    objects laid out the same way on the same drives.
    """
    keys = [f"{prefix}/{size_bytes}/{i:05d}.bin" for i in range(count)]
    missing = []
    for k in keys:
        try:
            if store.stat(k) != size_bytes:
                missing.append(k)
        except Exception:
            missing.append(k)
    if not missing:
        return keys

    print(f"    writing {len(missing)} object(s) of {size_bytes / 2**20:.0f} MiB", flush=True)
    # Distinct content per object so a benchmark cannot be satisfied from a single
    # cached buffer somewhere in the stack.
    src = torch.empty(size_bytes, dtype=torch.uint8, device=device)
    for i, k in enumerate(missing):
        src.fill_((i * 37 + 11) & 0xFF)
        torch.cuda.synchronize()
        store.put_from(k, src, size_bytes)
    del src
    torch.cuda.empty_cache()
    return keys


def run_cell(
    store,
    keys: list[str],
    size_bytes: int,
    concurrency: int,
    iters: int,
    device: str,
    servers: list[str],
    nics: list[str],
) -> dict:
    """One (size, concurrency) measurement."""
    # One destination buffer per concurrent stream, allocated up front: allocation
    # is not part of what we are measuring, and RDMA registration is per-buffer.
    dsts = [torch.empty(size_bytes, dtype=torch.uint8, device=device) for _ in range(concurrency)]
    torch.cuda.synchronize()

    total_ops = iters * concurrency
    lat: list[float] = []

    def one(slot: int, i: int) -> float:
        k = keys[(slot * iters + i) % len(keys)]
        t0 = time.perf_counter()
        n = store.get_into(k, dsts[slot], size_bytes)
        dt = time.perf_counter() - t0
        if n != size_bytes:
            raise RuntimeError(f"short read on {k}: {n} != {size_bytes}")
        return dt

    # A worker thread must bind the CUDA context and build its own client before
    # its first transfer -- see RdmaStore.prepare_thread. Doing it in the pool
    # initializer keeps that cost (and, for RDMA, a hard crash if skipped) out of
    # the measured window.
    def init_thread() -> None:
        store.prepare_thread()

    if concurrency == 1:
        store.prepare_thread()
        pool = None
    else:
        pool = ThreadPoolExecutor(
            max_workers=concurrency, initializer=init_thread, thread_name_prefix="fetch"
        )

    def stream(slot: int) -> list[float]:
        """Run one stream's whole sequence of fetches, on one thread, one buffer."""
        return [one(slot, i) for i in range(iters)]

    try:
        # Warm up every stream: first touch pays RDMA memory registration for that
        # buffer and TCP/RDMA connection setup, which would otherwise be charged to
        # the first measured iteration.
        if pool is None:
            one(0, 0)
        else:
            list(pool.map(lambda s: one(s, 0), range(concurrency)))

        nic0 = nic_counters(nics)
        hw0 = {d: rdma_hw_counters(d) for d in ("mlx5_0", "mlx5_1")}

        with RDMAWitness(servers) as witness, CPUSampler() as cpu:
            t_start = time.perf_counter()
            if pool is None:
                lat.extend(stream(0))
            else:
                # Exactly ONE task per stream, each looping over its own slot.
                #
                # Submitting slots x iters tasks instead would let the pool run two
                # tasks with the same slot on two threads at once, i.e. two
                # concurrent RDMA transfers into the same destination buffer. That
                # is not a benchmark artefact to shrug at: cuObject registers the
                # buffer by address, and concurrent register/deregister of one
                # address from two clients segfaults the process.
                futs = [pool.submit(stream, s) for s in range(concurrency)]
                for f in as_completed(futs):
                    lat.extend(f.result())
            wall = time.perf_counter() - t_start
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    nic1 = nic_counters(nics)
    hw1 = {d: rdma_hw_counters(d) for d in ("mlx5_0", "mlx5_1")}

    total_bytes = total_ops * size_bytes
    out = {
        "size_bytes": size_bytes,
        "concurrency": concurrency,
        "iters_per_stream": iters,
        "ops": total_ops,
        "bytes": total_bytes,
        "wall_s": wall,
        "throughput_gbps_bytes": total_bytes / wall / 1e9,
        "throughput_gib_s": total_bytes / wall / 2**30,
        "ops_per_s": total_ops / wall,
        "latency": _percentiles(lat),
        **cpu.as_dict(),
        # Floor the divisor at a tenth of a core: below that the /proc/stat sample
        # is mostly quantisation noise, and dividing by it yields meaningless
        # nine-digit "efficiency" figures.
        "gb_per_cpu_core": (
            (total_bytes / wall / 1e9) / cpu.cores_used if cpu.cores_used >= 0.1 else None
        ),
        "cpu_below_measurement_floor": cpu.cores_used < 0.1,
        "nic_rx_bytes_delta": {
            k: nic1.get(k, 0) - nic0.get(k, 0) for k in nic0 if k.endswith("rx_bytes")
        },
        "rdma_witness": witness.as_dict(),
        # Rising drop/retry counters mean the fabric is not lossless, which caps
        # RDMA throughput; report them with the result rather than after the fact.
        "fabric_health": {
            dev: {
                c: hw1[dev].get(c, 0) - hw0[dev].get(c, 0)
                for c in ("out_of_buffer", "packet_seq_err", "local_ack_timeout_err",
                          "rnr_nak_retry_err", "req_cqe_error", "resp_cqe_error")
                if c in hw1.get(dev, {})
            }
            for dev in hw1
        },
    }

    del dsts
    torch.cuda.empty_cache()
    return out


# ------------------------------------------------------------ multi-process ---
#
# A single CPython process caps the HTTP path at roughly 2 GB/s no matter how many
# threads it runs -- measured identically for minio-py, bare urllib3, and even a
# raw socket with recv_into, while the same fabric does 41.5 GB/s under iperf3 and
# 39.4 GB/s under warp (native Go). The limit is the process, not the transport.
#
# Real PyTorch loaders sidestep this with DataLoader worker *processes*, so the
# honest HTTP baseline is multi-process too. Reporting only the single-process
# number would overstate RDMA's advantage; reporting only the multi-process number
# would hide how much machinery HTTP needs to get there. We measure both.


def _mp_worker(payload: dict) -> dict:
    """Run one process's share of a cell. Must be importable for spawn."""
    import torch as _torch

    from .s3 import StoreConfig as _Cfg, make_store as _make

    _torch.cuda.set_device(_torch.device(payload["device"]))
    cfg = _Cfg(**payload["cfg"])
    store = _make(payload["backend"], cfg, local_root=payload["local_root"])
    try:
        store.prepare_thread()
        keys = payload["keys"]
        size = payload["size_bytes"]
        conc = payload["concurrency"]
        iters = payload["iters"]

        dsts = [_torch.empty(size, dtype=_torch.uint8, device=payload["device"]) for _ in range(conc)]
        _torch.cuda.synchronize()

        def one(slot: int, i: int) -> float:
            k = keys[(slot * iters + i) % len(keys)]
            t0 = time.perf_counter()
            n = store.get_into(k, dsts[slot], size)
            if n != size:
                raise RuntimeError(f"short read {k}: {n} != {size}")
            return time.perf_counter() - t0

        def stream(slot: int) -> list[float]:
            return [one(slot, i) for i in range(iters)]

        if conc == 1:
            one(0, 0)
            t0 = time.perf_counter()
            lat = stream(0)
            wall = time.perf_counter() - t0
        else:
            with ThreadPoolExecutor(
                max_workers=conc, initializer=store.prepare_thread
            ) as pool:
                list(pool.map(lambda s: one(s, 0), range(conc)))
                t0 = time.perf_counter()
                futs = [pool.submit(stream, s) for s in range(conc)]
                lat = [x for f in as_completed(futs) for x in f.result()]
                wall = time.perf_counter() - t0

        return {"bytes": conc * iters * size, "wall_s": wall, "lat": lat}
    finally:
        store.close()


def run_cell_mp(
    cfg_dict: dict,
    backend: str,
    local_root: str,
    keys: list[str],
    size_bytes: int,
    concurrency: int,
    iters: int,
    processes: int,
    device: str,
    servers: list[str],
    nics: list[str],
) -> dict:
    """A cell run across ``processes`` worker processes, each with ``concurrency`` threads."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor as _PPE

    # spawn, not fork: a forked child inherits a CUDA context it cannot use.
    ctx = mp.get_context("spawn")

    # Spread workers across GPUs round-robin. Two reasons: it mirrors what a real
    # DDP job looks like (one process per GPU), and packing every worker's
    # destination buffers onto one device runs out of VRAM fast --
    # processes x concurrency x size is easily hundreds of GiB.
    ngpu = max(1, torch.cuda.device_count())
    base = torch.device(device).index or 0

    # Disjoint key slices so processes do not all read the same objects.
    per = max(1, len(keys) // processes)
    payloads = [
        {
            "cfg": cfg_dict,
            "backend": backend,
            "local_root": local_root,
            "keys": keys[i * per : (i + 1) * per] or keys,
            "size_bytes": size_bytes,
            "concurrency": concurrency,
            "iters": iters,
            "device": f"cuda:{(base + i) % ngpu}",
        }
        for i in range(processes)
    ]

    per_gpu = {}
    for pl in payloads:
        per_gpu[pl["device"]] = per_gpu.get(pl["device"], 0) + concurrency * size_bytes
    worst = max(per_gpu.values()) if per_gpu else 0
    if worst > 0.8 * torch.cuda.get_device_properties(0).total_memory:
        raise RuntimeError(
            f"destination buffers would need {worst / 2**30:.1f} GiB on one GPU "
            f"(processes={processes} x concurrency={concurrency} x {size_bytes / 2**20:.0f} MiB); "
            "lower --processes/--concurrency or use more GPUs"
        )

    nic0 = nic_counters(nics)
    hw0 = {d: rdma_hw_counters(d) for d in ("mlx5_0", "mlx5_1")}

    with RDMAWitness(servers) as witness, CPUSampler() as cpu:
        t_start = time.perf_counter()
        # ProcessPoolExecutor, not multiprocessing.Pool: if a worker dies abruptly
        # (a segfault inside the RDMA client, say) Pool.map blocks forever with no
        # diagnostic, whereas the executor raises BrokenProcessPool. A benchmark
        # that hangs silently on a crash is worse than one that fails.
        with _PPE(max_workers=processes, mp_context=ctx) as pool:
            parts = list(pool.map(_mp_worker, payloads))
        wall = time.perf_counter() - t_start

    nic1 = nic_counters(nics)
    hw1 = {d: rdma_hw_counters(d) for d in ("mlx5_0", "mlx5_1")}

    total_bytes = sum(p["bytes"] for p in parts)
    total_ops = processes * concurrency * iters
    lat = [x for p in parts for x in p["lat"]]

    return {
        "size_bytes": size_bytes,
        "concurrency": concurrency,
        "processes": processes,
        "gpu_bytes_per_device": per_gpu,
        "iters_per_stream": iters,
        "ops": total_ops,
        "bytes": total_bytes,
        # Wall clock includes process startup and CUDA-context creation, which is
        # real cost for a short cell; per-process walls are kept so the steady-state
        # rate can be read separately.
        "wall_s": wall,
        "worker_wall_s": [p["wall_s"] for p in parts],
        "throughput_gbps_bytes": total_bytes / wall / 1e9,
        "throughput_gbps_steady": total_bytes / max(p["wall_s"] for p in parts) / 1e9,
        "ops_per_s": total_ops / wall,
        "latency": _percentiles(lat),
        **cpu.as_dict(),
        "gb_per_cpu_core": (
            (total_bytes / wall / 1e9) / cpu.cores_used if cpu.cores_used >= 0.1 else None
        ),
        "nic_rx_bytes_delta": {
            k: nic1.get(k, 0) - nic0.get(k, 0) for k in nic0 if k.endswith("rx_bytes")
        },
        "rdma_witness": witness.as_dict(),
        "fabric_health": {
            dev: {
                c: hw1[dev].get(c, 0) - hw0[dev].get(c, 0)
                for c in ("out_of_buffer", "packet_seq_err", "local_ack_timeout_err",
                          "rnr_nak_retry_err", "req_cqe_error", "resp_cqe_error")
                if c in hw1.get(dev, {})
            }
            for dev in hw1
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="B1: raw object GET throughput into GPU memory")
    ap.add_argument("--backend", required=True, choices=["http", "rdma", "local"])
    ap.add_argument("--bucket", default="bench-fetch")
    ap.add_argument("--prefix", default="b1")
    ap.add_argument("--endpoint", default=None)
    ap.add_argument("--endpoints", default=None,
                    help="comma-separated; HTTP spreads connections over all of them")
    ap.add_argument("--sizes", default="1,4,16,64,256",
                    help="object sizes in MiB")
    ap.add_argument("--concurrency", default="1,4,8,16,32")
    ap.add_argument("--iters", type=int, default=8, help="fetches per stream per cell")
    ap.add_argument("--processes", type=int, default=1,
                    help="worker processes; >1 models a real DataLoader and is the only "
                         "way the Python HTTP path gets past its ~2 GB/s per-process cap")
    ap.add_argument("--objects", type=int, default=64, help="distinct objects per size")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--servers", default="aistor1:9000,aistor2:9000")
    ap.add_argument("--nics", default="enp27s0np0,enp157s0np0")
    ap.add_argument("--http-parts", type=int, default=8)
    ap.add_argument("--http-threads", type=int, default=32)
    ap.add_argument("--local-root", default="")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)

    sizes = [int(s) * (1 << 20) for s in args.sizes.split(",") if s.strip()]
    concs = [int(c) for c in args.concurrency.split(",") if c.strip()]
    servers = [s.strip() for s in args.servers.split(",") if s.strip()]
    nics = [n.strip() for n in args.nics.split(",") if n.strip()]
    eps = [e.strip() for e in args.endpoints.split(",")] if args.endpoints else []

    cfg = StoreConfig(bucket=args.bucket, http_parts=args.http_parts, http_threads=args.http_threads)
    if args.endpoint:
        cfg.endpoint = args.endpoint
    if eps:
        cfg.endpoints = eps

    torch.cuda.set_device(torch.device(args.device))

    # Objects are always created over HTTP so the dataset is identical regardless
    # of which transport is being measured.
    from minio import Minio

    admin = Minio(cfg.endpoint, cfg.access_key, cfg.secret_key, secure=cfg.secure)
    if not admin.bucket_exists(args.bucket):
        admin.make_bucket(args.bucket)

    print(f"== B1 fetch throughput | backend={args.backend} ==")
    print(f"   endpoint(s): {cfg.all_endpoints()}")
    print(f"   sizes: {[s // 2**20 for s in sizes]} MiB   concurrency: {concs}"
          f"   processes: {args.processes}")

    writer = make_store("http", cfg)
    prepared: dict[int, list[str]] = {}
    for sz in sizes:
        print(f"   preparing {sz // 2**20} MiB objects", flush=True)
        prepared[sz] = prepare_objects(writer, args.prefix, sz, args.objects, args.device)
    writer.close()

    cfg_dict = {
        "endpoint": cfg.endpoint, "access_key": cfg.access_key, "secret_key": cfg.secret_key,
        "bucket": cfg.bucket, "secure": cfg.secure, "region": cfg.region,
        "http_parts": cfg.http_parts, "http_threads": cfg.http_threads,
        "http_chunk": cfg.http_chunk, "endpoints": cfg.endpoints,
    }
    store = None if args.processes > 1 else make_store(args.backend, cfg, local_root=args.local_root)

    cells = []
    hdr = f"{'size':>8} {'conc':>5} {'GB/s':>8} {'ops/s':>9} {'p50 ms':>9} {'p99 ms':>9} {'cores':>7} {'GB/s/core':>10}"
    print("\n" + hdr)
    print("-" * len(hdr))

    for sz in sizes:
        for c in concs:
            try:
                if args.processes > 1:
                    cell = run_cell_mp(
                        cfg_dict, args.backend, args.local_root, prepared[sz], sz, c,
                        args.iters, args.processes, args.device, servers, nics,
                    )
                else:
                    cell = run_cell(store, prepared[sz], sz, c, args.iters, args.device, servers, nics)
            except Exception as exc:
                print(f"{sz // 2**20:>7}M {c:>5}   FAILED: {type(exc).__name__}: {exc}")
                cells.append({"size_bytes": sz, "concurrency": c, "error": f"{type(exc).__name__}: {exc}"})
                continue

            # Proof-of-RDMA. Without this the whole comparison is worthless: a
            # silent fallback would show up as "RDMA is no faster than HTTP".
            if args.backend == "rdma":
                got = cell["rdma_witness"]["rdma_read_bytes"] or 0
                if got < cell["bytes"] * 0.5:
                    print(f"{sz // 2**20:>7}M {c:>5}   FAILED: RDMA not used "
                          f"(server counted {got:,.0f} of {cell['bytes']:,} bytes)")
                    cell["error"] = "rdma-not-used"
                    cells.append(cell)
                    continue

            lat = cell["latency"]
            eff = cell["gb_per_cpu_core"]
            eff_s = f"{eff:>10.2f}" if eff is not None else f"{'<floor':>10}"
            print(f"{sz // 2**20:>7}M {c:>5} {cell['throughput_gbps_bytes']:>8.2f} "
                  f"{cell['ops_per_s']:>9.1f} {lat['p50_ms']:>9.2f} {lat['p99_ms']:>9.2f} "
                  f"{cell['cpu_cores_used']:>7.2f} {eff_s}", flush=True)
            cells.append(cell)

    if store is not None:
        store.close()

    ok = [c for c in cells if "error" not in c]
    best = max(ok, key=lambda c: c["throughput_gbps_bytes"], default=None)

    rec = RunRecord(
        benchmark="b1-fetch" + (f"-{args.tag}" if args.tag else ""),
        backend=args.backend,
        params={
            "sizes_mib": [s // 2**20 for s in sizes],
            "concurrency": concs,
            "iters_per_stream": args.iters,
            "objects_per_size": args.objects,
            "bucket": args.bucket,
            "endpoints": cfg.all_endpoints(),
            "http_parts": args.http_parts,
            "http_threads": args.http_threads,
            "processes": args.processes,
            "device": args.device,
        },
        results={
            "cells": cells,
            "peak_gbps_bytes": best["throughput_gbps_bytes"] if best else None,
            "peak_cell": {"size_bytes": best["size_bytes"], "concurrency": best["concurrency"]} if best else None,
            "failures": len(cells) - len(ok),
        },
        environment={
            "gpus": gpu_info(),
            "torch": torch.__version__,
            "cpu_count": CPUSampler().ncpu,
        },
    )
    path = rec.save(args.results_dir)
    print(f"\npeak: {best['throughput_gbps_bytes']:.2f} GB/s at "
          f"{best['size_bytes'] // 2**20} MiB x {best['concurrency']}" if best else "\nno successful cells")
    print(f"saved: {path}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
