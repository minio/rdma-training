#!/usr/bin/env python3
# Copyright 2026 MinIO, Inc.
# SPDX-License-Identifier: Apache-2.0

"""B7 — LLM pretraining data path: tokenized int32 shards.

This benchmark exists to document a negative result rather than to sell a
positive one. Tokenized text is GPU-native (int32 ids), so RDMA *does* apply to
it — unlike JPEG. But the data path is so cheap that the transport is irrelevant:

    LLM pretraining reads ~4 bytes per token and spends ~6N FLOPs on it, so a
    dense 8B model on 8x H200 wants ~0.3 MB/s of storage. ResNet-50 on the same
    GPUs wants ~4,800 MB/s.

Four orders of magnitude. So we measure what a token loader can actually deliver,
convert it into "how many H200s' worth of a given model could this feed", and let
the ratio speak.

    python -m s3rdma_train.bench_tokens --backend rdma --bucket llm-tokens

Reports tokens/s, GB/s, CPU cores, and the implied model scale the loader could
sustain. Every RDMA run is bracketed by RDMAWitness, as everywhere else here.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

import torch

from .metrics import CPUSampler, RDMAWitness, RunRecord, gpu_info
from .s3 import StoreConfig, make_store

# H200 bf16 dense peak, and the MFU a well-tuned pretraining job actually reaches.
H200_BF16_PEAK_FLOPS = 989e12
ASSUMED_MFU = 0.45
# Transformer training cost per token: forward + backward ~= 6 * active params.
FLOPS_PER_PARAM_PER_TOKEN = 6


def tokens_per_s_per_gpu(active_params: float) -> float:
    return H200_BF16_PEAK_FLOPS * ASSUMED_MFU / (FLOPS_PER_PARAM_PER_TOKEN * active_params)


# (label, active params) — active is what sets FLOPs; for MoE that is far below total.
MODEL_SCALES = [
    ("Llama-3 8B (dense)", 8e9),
    ("Llama-3 70B (dense)", 70e9),
    ("DeepSeek-V3-class MoE (37B active of 671B)", 37e9),
    ("405B (dense)", 405e9),
]


class TokenShardLoader:
    """Read fixed-size int32 token shards, yielding [batch, seq_len] int32 on GPU.

    Deliberately the same structure as the image loaders: fetch a whole shard into
    a pooled, address-stable device buffer, then slice batches out of it with no
    host involvement. Sequences are fixed length, so no index is needed —
    sequence i begins at i * seq_len * 4.
    """

    def __init__(self, store, keys: list[str], batch_seqs: int, device: str = "cuda:0",
                 fetch_workers: int = 4) -> None:
        self.store = store
        self.keys = keys
        self.batch_seqs = batch_seqs
        self.device = torch.device(device)
        self.fetch_workers = fetch_workers
        self._meta: dict[str, dict] = {}
        self.bytes_fetched = 0
        self.tokens_yielded = 0
        self.shards = 0
        self.fetch_wait_s = 0.0

    def _meta_for(self, key: str) -> dict:
        m = self._meta.get(key)
        if m is None:
            m = self._meta[key] = json.loads(
                self.store.get_bytes(key.rsplit(".", 1)[0] + ".json").decode()
            )
        return m

    def __iter__(self):
        from .dataset import _BufferPool, _Prefetcher

        m0 = self._meta_for(self.keys[0])
        seq_len = m0["seq_len"]
        shard_bytes = m0["sequences"] * seq_len * 4
        pool = _BufferPool(shard_bytes, self.fetch_workers, self.device, per_worker=2)

        def fetch(key: str, worker: int = 0):
            m = self._meta_for(key)
            nb = m["sequences"] * m["seq_len"] * 4
            buf = pool.acquire(worker)
            try:
                got = self.store.get_into(key, buf, nb)
            except BaseException:
                pool.release(worker, buf)
                raise
            if got != nb:
                pool.release(worker, buf)
                raise RuntimeError(f"{key}: short read {got} != {nb}")
            return worker, buf, m, nb

        pf = _Prefetcher(fetch, self.keys, depth=max(2, self.fetch_workers),
                         workers=self.fetch_workers,
                         on_thread_start=self.store.prepare_thread).start()
        stream = pf.__iter__()
        try:
            while True:
                t0 = time.perf_counter()
                try:
                    worker, buf, m, nb = next(stream)
                except StopIteration:
                    break
                self.fetch_wait_s += time.perf_counter() - t0
                self.shards += 1
                self.bytes_fetched += nb
                try:
                    # Reinterpret the shard as [sequences, seq_len] int32 in place.
                    toks = buf[:nb].view(torch.int32).view(m["sequences"], m["seq_len"])
                    for s in range(0, m["sequences"], self.batch_seqs):
                        e = min(s + self.batch_seqs, m["sequences"])
                        if e - s < self.batch_seqs:
                            break
                        self.tokens_yielded += (e - s) * m["seq_len"]
                        yield toks[s:e]
                finally:
                    del toks
                    pool.release(worker, buf)
        finally:
            pf.stop()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="B7: LLM token data path")
    ap.add_argument("--backend", required=True, choices=["http", "rdma", "local"])
    ap.add_argument("--bucket", default="llm-tokens")
    ap.add_argument("--prefix", default="train")
    ap.add_argument("--endpoints", default="aistor1:9000,aistor2:9000")
    ap.add_argument("--servers", default="aistor1:9000,aistor2:9000")
    ap.add_argument("--batch-seqs", type=int, default=64, help="sequences per batch")
    # Measured by SHARDS, not steps. A batch costs ~0.02 ms here (it is a view
    # into an already-resident shard), so any step budget small enough to finish
    # quickly is also small enough that the prefetcher fetched every shard before
    # the measurement window opened -- which reads as "0 RDMA bytes, 16 billion
    # tokens/s". Counting shards ties the window to actual transfers.
    ap.add_argument("--warmup-shards", type=int, default=2)
    ap.add_argument("--shards-measured", type=int, default=16)
    ap.add_argument("--fetch-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--local-root", default="")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)

    servers = [s.strip() for s in args.servers.split(",") if s.strip()]
    cfg = StoreConfig(bucket=args.bucket)
    cfg.endpoints = [e.strip() for e in args.endpoints.split(",") if e.strip()]
    cfg.endpoint = cfg.endpoints[0]
    torch.cuda.set_device(torch.device(args.device))

    store = make_store(args.backend, cfg, local_root=args.local_root)
    keys = sorted(k for k, _ in store.list(args.prefix) if k.endswith(".bin"))
    if not keys:
        raise RuntimeError(f"no .bin shards under {args.bucket}/{args.prefix}")

    # Cycle shards so a long run does not exhaust the dataset.
    loader = TokenShardLoader(store, keys * 64, args.batch_seqs, device=args.device,
                              fetch_workers=args.fetch_workers)
    it = iter(loader)

    print(f"== B7 LLM token data path | backend={args.backend} ==")
    print(f"   {len(keys)} shards, batch={args.batch_seqs} seqs, "
          f"fetch_workers={args.fetch_workers}")

    base_tokens = base_bytes = 0
    base_wait = 0.0
    t0 = None
    started = False
    witness, cpu = RDMAWitness(servers), CPUSampler()
    step_ms: list[float] = []

    while True:
        if not started and loader.shards >= args.warmup_shards:
            torch.cuda.synchronize()
            base_tokens, base_bytes = loader.tokens_yielded, loader.bytes_fetched
            base_wait = loader.fetch_wait_s
            base_shards = loader.shards
            witness.__enter__(); cpu.__enter__()
            t0 = time.perf_counter()
            started = True
        if started and loader.shards - base_shards >= args.shards_measured:
            break
        ts = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            break
        # Touch the batch so the fetch is not optimised away.
        _ = int(batch[0, 0])
        if started:
            step_ms.append((time.perf_counter() - ts) * 1e3)

    torch.cuda.synchronize()
    wall = time.perf_counter() - t0 if t0 else 0.0
    cpu.__exit__(None, None, None); witness.__exit__(None, None, None)
    it.close()

    tokens = loader.tokens_yielded - base_tokens
    nbytes = loader.bytes_fetched - base_bytes
    wait = loader.fetch_wait_s - base_wait
    tok_s = tokens / wall if wall else 0.0
    gbps = nbytes / wall / 1e9 if wall else 0.0

    print(f"\n   tokens/s          : {tok_s:,.0f}")
    print(f"   storage           : {nbytes / 2**30:.1f} GiB @ {gbps:.2f} GB/s")
    print(f"   host CPU          : {cpu.cores_used:.2f} cores")
    print(f"   blocked on storage: {wait:.2f} s ({100 * wait / wall if wall else 0:.1f}%)")
    print(f"   server RDMA bytes : {witness.delta.read_bytes:,.0f}")
    print(f"   median step       : {statistics.median(step_ms) if step_ms else 0:.3f} ms")

    print(f"\n   This loader could feed (at {ASSUMED_MFU:.0%} MFU on H200):")
    implied = {}
    for label, active in MODEL_SCALES:
        per_gpu = tokens_per_s_per_gpu(active)
        gpus = tok_s / per_gpu
        implied[label] = {"tokens_s_per_gpu": per_gpu, "gpus_feedable": gpus}
        print(f"     {label:<44} {gpus:>9,.0f} H200s  "
              f"({per_gpu:,.0f} tok/s/GPU)")

    if args.backend == "rdma":
        got = witness.delta.read_bytes or 0
        if got < nbytes * 0.5:
            print("\n   WARNING: server RDMA counters did not account for these bytes;"
                  "\n            this run silently used HTTP.")

    rec = RunRecord(
        benchmark="b7-tokens" + (f"-{args.tag}" if args.tag else ""),
        backend=args.backend,
        params={"bucket": args.bucket, "batch_seqs": args.batch_seqs,
                "warmup_shards": args.warmup_shards,
                "shards_measured": args.shards_measured,
                "fetch_workers": args.fetch_workers, "endpoints": cfg.endpoints,
                "assumed_mfu": ASSUMED_MFU},
        results={"tokens_per_s": tok_s, "gbps": gbps, "bytes": nbytes,
                 "cpu_cores": cpu.cores_used, "fetch_wait_s": wait,
                 "wall_s": wall, "shards_in_window": loader.shards - base_shards,
                 "median_step_ms": statistics.median(step_ms) if step_ms else None,
                 "implied_gpu_capacity": implied},
        environment={"gpus": gpu_info(), "torch": torch.__version__},
        rdma_witness=witness.as_dict(),
    )
    print(f"\nsaved: {rec.save(args.results_dir)}")
    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
