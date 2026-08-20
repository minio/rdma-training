# Copyright 2026 MinIO, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shard format: a WebDataset-compatible tar plus a sidecar byte index.

Why a sidecar index exists
--------------------------
The point of GPU-Direct RDMA is that a shard's bytes land in VRAM without ever
touching host memory. But a tar is self-describing: to find the samples you must
read 512-byte headers, and you cannot read headers that live in VRAM without
copying them back to the host — which would give back exactly the copy RDMA saved.

So shard-building writes, alongside each ``train-000000.tar``, a small
``train-000000.index.json`` recording every sample's byte offset, length and
label. At training time the flow is:

    fetch index (once, tiny, over HTTP)
    RDMA whole tar -> uint8 CUDA tensor
    for each sample: buf[off : off+len]      # a device-side view, no copy
    torchvision.io.decode_jpeg(views, device="cuda")   # nvJPEG, batched

The tar itself stays a normal tar: `tar tf` works, and any standard WebDataset
reader can consume it. The index is an optimisation, not a new container format.

Layout inside a shard, per sample::

    <key>.jpg     the encoded JPEG
    <key>.cls     the class index as ASCII, for WebDataset interoperability

Offsets in the index point at the ``.jpg`` payload only.
"""

from __future__ import annotations

import io
import json
import tarfile
from dataclasses import dataclass, asdict
from typing import Iterable, Optional

# A ustar header is exactly one 512-byte block, so a member's payload starts 512
# bytes after its header -- but only while the name fits the ustar name field and
# no PAX extension header is emitted. Shard building asserts both, which is what
# makes the arithmetic below sound rather than hopeful.
TAR_BLOCK = 512
USTAR_MAX_NAME = 99


@dataclass
class Sample:
    key: str
    offset: int  # byte offset of the JPEG payload within the tar
    length: int  # JPEG length in bytes
    label: int

    @staticmethod
    def from_dict(d: dict) -> "Sample":
        return Sample(d["key"], int(d["offset"]), int(d["length"]), int(d["label"]))


@dataclass
class ShardIndex:
    shard: str  # object key of the tar
    size: int  # total tar size in bytes
    samples: list[Sample]

    @property
    def num_samples(self) -> int:
        return len(self.samples)

    @property
    def max_sample_length(self) -> int:
        return max((s.length for s in self.samples), default=0)

    def to_json(self) -> bytes:
        return json.dumps(
            {"shard": self.shard, "size": self.size, "samples": [asdict(s) for s in self.samples]},
            separators=(",", ":"),
        ).encode()

    @staticmethod
    def from_json(raw: bytes) -> "ShardIndex":
        d = json.loads(raw)
        return ShardIndex(
            shard=d["shard"],
            size=int(d["size"]),
            samples=[Sample.from_dict(s) for s in d["samples"]],
        )

    @staticmethod
    def key_for(shard_key: str) -> str:
        """Index object key for a shard key."""
        base = shard_key[:-4] if shard_key.endswith(".tar") else shard_key
        return base + ".index.json"


class ShardWriter:
    """Build one tar shard in memory and record each sample's byte range.

    In-memory rather than streamed to disk because the shard then uploads straight
    from the buffer, and a benchmark rig has RAM to spare. A 1 GiB shard is the
    default target: large enough that per-object overhead disappears, small enough
    that a worker's buffer is not unwieldy.
    """

    def __init__(self, shard_key: str) -> None:
        self.shard_key = shard_key
        self._buf = io.BytesIO()
        # USTAR (not the default PAX): PAX writes an extra header block per member
        # for long names or big sizes, which would break the "+512" arithmetic.
        self._tar = tarfile.open(fileobj=self._buf, mode="w", format=tarfile.USTAR_FORMAT)
        self._samples: list[Sample] = []

    def add(self, key: str, jpeg: bytes, label: int) -> None:
        jpg_name = f"{key}.jpg"
        cls_name = f"{key}.cls"
        if len(jpg_name) > USTAR_MAX_NAME or len(cls_name) > USTAR_MAX_NAME:
            raise ValueError(f"member name too long for ustar: {jpg_name!r}")

        header_off = self._buf.tell()
        ti = tarfile.TarInfo(jpg_name)
        ti.size = len(jpeg)
        ti.mode = 0o644
        # Fixed mtime/uid/gid so a rebuild is byte-identical, which makes shards
        # comparable across runs.
        ti.mtime = 0
        ti.uid = ti.gid = 0
        ti.uname = ti.gname = ""
        self._tar.addfile(ti, io.BytesIO(jpeg))
        data_off = header_off + TAR_BLOCK

        # Cheap invariant: the payload really is where we said it is.
        end = self._buf.tell()
        pad = -len(jpeg) % TAR_BLOCK
        if end != data_off + len(jpeg) + pad:
            raise AssertionError(
                f"tar layout mismatch for {jpg_name}: header={header_off} "
                f"data={data_off} len={len(jpeg)} end={end}"
            )

        cls = str(label).encode()
        cti = tarfile.TarInfo(cls_name)
        cti.size = len(cls)
        cti.mode = 0o644
        cti.mtime = 0
        cti.uid = cti.gid = 0
        cti.uname = cti.gname = ""
        self._tar.addfile(cti, io.BytesIO(cls))

        self._samples.append(Sample(key=key, offset=data_off, length=len(jpeg), label=label))

    @property
    def size(self) -> int:
        """Current uncompressed size, excluding the not-yet-written end blocks."""
        return self._buf.tell()

    @property
    def num_samples(self) -> int:
        return len(self._samples)

    def finish(self) -> tuple[memoryview, ShardIndex]:
        self._tar.close()  # writes the two zero end-of-archive blocks
        data = self._buf.getbuffer()
        index = ShardIndex(shard=self.shard_key, size=len(data), samples=self._samples)
        return data, index


def verify_index(tar_bytes: bytes, index: ShardIndex, checks: int = 8) -> None:
    """Spot-check that indexed ranges really are JPEGs.

    Cheap insurance against an off-by-512 in the offset arithmetic, which would
    otherwise surface much later as an inscrutable nvJPEG decode failure.
    """
    n = len(index.samples)
    if n == 0:
        return
    step = max(1, n // max(1, checks))
    for s in index.samples[::step]:
        blob = tar_bytes[s.offset : s.offset + s.length]
        if len(blob) != s.length:
            raise AssertionError(f"{s.key}: truncated range")
        if blob[:2] != b"\xff\xd8":
            raise AssertionError(
                f"{s.key}: range at offset {s.offset} does not start with a JPEG SOI marker"
            )


def iter_batches(samples: Iterable[Sample], batch_size: int) -> Iterable[list[Sample]]:
    batch: list[Sample] = []
    for s in samples:
        batch.append(s)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
