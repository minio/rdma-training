#!/usr/bin/env python3
# Copyright 2026 MinIO, Inc.
# SPDX-License-Identifier: Apache-2.0

"""B3 — ResNet-50 training: how long do N batches take, per storage backend?

This is the headline the whole project exists to answer, and it is deliberately
reported as a breakdown rather than a single number, because a single number hides
the thing that decides whether storage matters:

    step time = fetch_wait + augment/decode + forward/backward/optimizer

If ``fetch_wait`` is near zero, the job is compute-bound and a faster transport
cannot make it faster — no matter how much faster the transport is in isolation.
Reporting only "seconds for N batches" would let a compute-bound result be read as
"RDMA doesn't work". Reporting the breakdown says *why*.

Layouts, and which transports each can use:

``--layout raw`` (GPU-native uint8 shards)
    ``--backend rdma`` and ``--backend http`` both work. This is the apples-to-apples
    storage comparison.

``--layout jpeg`` (JPEG tar shards)
    ``--backend http`` only. RDMA needs a device destination; nvJPEG needs a host
    source. See plans/reports/02-where-rdma-applies.md.

Single-GPU by default. ``--gpus N`` runs N processes in DDP, which is what makes the
aggregate storage demand large enough to matter.

    python -m s3rdma_train.train_resnet50 --backend rdma --layout raw --steps 200
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn

from .dataset import JpegShardLoader, RawShardLoader
from .metrics import CPUSampler, RDMAWitness, RunRecord, gpu_info, rdma_hw_counters
from .s3 import StoreConfig, make_store


def build_model(device: torch.device, channels_last: bool = True) -> nn.Module:
    import torchvision

    m = torchvision.models.resnet50(weights=None).to(device)
    if channels_last:
        m = m.to(memory_format=torch.channels_last)
    return m


def list_shards(store, prefix: str, layout: str, limit: int = 0) -> list[str]:
    suffix = ".raw" if layout == "raw" else ".tar"
    keys = sorted(k for k, _ in store.list(prefix) if k.endswith(suffix))
    if limit:
        keys = keys[:limit]
    return keys


def run(args, rank: int, world: int) -> dict:
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True
    # TF32 on: this is what anyone training ResNet-50 on Hopper would do, and
    # leaving it off would inflate compute time and flatter the storage layer.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cfg = StoreConfig(bucket=args.bucket)
    if args.endpoints:
        cfg.endpoints = [e.strip() for e in args.endpoints.split(",") if e.strip()]
        cfg.endpoint = cfg.endpoints[0]
    cfg.http_parts = args.http_parts
    cfg.http_threads = args.http_threads

    store = make_store(args.backend, cfg, local_root=args.local_root)
    keys = list_shards(store, args.prefix, args.layout, args.max_shards)
    if not keys:
        raise RuntimeError(f"no {args.layout} shards under {args.bucket}/{args.prefix}")
    # Disjoint shards per rank: two ranks reading the same shard would both inflate
    # cache hits and duplicate samples within a step.
    mine = keys[rank::world]
    if not mine:
        raise RuntimeError(f"rank {rank}: no shards after splitting {len(keys)} across {world}")

    Loader = RawShardLoader if args.layout == "raw" else JpegShardLoader
    loader = Loader(
        store, mine * args.epochs_over_shards, args.batch_size, device=str(device),
        crop=args.crop, prefetch=args.prefetch, seed=args.seed + rank,
        fetch_workers=args.fetch_workers,
    )
    loader.preload_metadata()

    if args.loader_only:
        model, opt, lossfn = None, None, None
    else:
        model = build_model(device, channels_last=not args.no_channels_last)
        if world > 1:
            model = nn.parallel.DistributedDataParallel(model, device_ids=[device.index])
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                              weight_decay=1e-4)
        lossfn = nn.CrossEntropyLoss()
    scaler_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16

    step_times: list[float] = []
    compute_times: list[float] = []
    losses: list[float] = []
    warmup_done_at = None

    servers = [s.strip() for s in args.servers.split(",") if s.strip()]
    fabric0 = {d: rdma_hw_counters(d) for d in ("mlx5_0", "mlx5_1")}

    it = iter(loader)
    step = 0
    base_bytes = base_shards = 0
    base_augment = base_decode = base_wait = 0.0
    total_steps = args.warmup + args.steps

    witness = RDMAWitness(servers)
    cpu = CPUSampler()
    t_wall0 = None

    while step < total_steps:
        try:
            x, y = next(it)
        except StopIteration:
            break

        if step == args.warmup:
            # Start measuring only after warmup: the first steps pay cudnn
            # autotuning, RDMA buffer registration and connection setup.
            #
            # Snapshot the loader's counters here too. Shards are fetched by a
            # background thread, and one 768 MiB shard feeds many steps, so shards
            # pulled during warmup are NOT part of the measured window. Without this
            # delta the proof check compares the server's RDMA bytes for the window
            # against bytes fetched for the whole run and wrongly reports a
            # fallback.
            torch.cuda.synchronize()
            base_bytes = loader.stats.bytes_fetched
            base_shards = loader.stats.shards
            base_augment = loader.stats.augment_s
            base_decode = loader.stats.decode_s
            base_wait = loader.stats.fetch_wait_s
            witness.__enter__()
            cpu.__enter__()
            t_wall0 = time.perf_counter()
            warmup_done_at = step

        t0 = time.perf_counter()
        if args.loader_only:
            # Touch the batch so the loader's work is not optimised away, but do no
            # model compute. What remains is the delivery rate.
            _ = float(x[0, 0, 0, 0])
            torch.cuda.synchronize()
            loss_val = 0.0
        else:
            with torch.autocast("cuda", dtype=scaler_dtype, enabled=args.amp != "off"):
                out = model(x)
                loss = lossfn(out, y)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            loss_val = float(loss.detach())
        dt = time.perf_counter() - t0

        if step >= args.warmup:
            compute_times.append(dt)
            losses.append(loss_val)
        step += 1

    torch.cuda.synchronize()
    wall = (time.perf_counter() - t_wall0) if t_wall0 else 0.0
    if warmup_done_at is not None:
        cpu.__exit__(None, None, None)
        witness.__exit__(None, None, None)

    # The loop exits on step count, not exhaustion, so the loader generator is still
    # suspended and its finally block -- which publishes the prefetch thread's
    # timings -- has not run. Close it explicitly or fetch_s stays 0.
    it.close()

    fabric1 = {d: rdma_hw_counters(d) for d in ("mlx5_0", "mlx5_1")}
    measured = len(compute_times)
    imgs = measured * args.batch_size

    st = loader.stats.as_dict()
    # Counters restricted to the measured window.
    window_bytes = loader.stats.bytes_fetched - base_bytes
    window = {
        "bytes_fetched": window_bytes,
        "shards": loader.stats.shards - base_shards,
        "fetch_wait_s": round(loader.stats.fetch_wait_s - base_wait, 4),
        "augment_s": round(loader.stats.augment_s - base_augment, 4),
        "decode_s": round(loader.stats.decode_s - base_decode, 4),
        "fetch_gbps_effective": (window_bytes / wall / 1e9) if wall > 0 else None,
    }
    out = {
        "rank": rank,
        "world": world,
        "steps_measured": measured,
        "batch_size": args.batch_size,
        "images": imgs,
        "wall_s": wall,
        "images_per_s": imgs / wall if wall > 0 else None,
        "sec_per_step_median": statistics.median(compute_times) if compute_times else None,
        "sec_per_step_mean": statistics.fmean(compute_times) if compute_times else None,
        "compute_s_total": sum(compute_times),
        "loader": st,
        "window": window,
        # The decisive ratio: how much of the wall clock was spent waiting for
        # storage rather than computing.
        "fetch_wait_fraction": (window["fetch_wait_s"] / wall) if wall > 0 else None,
        "compute_fraction": (sum(compute_times) / wall) if wall > 0 else None,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        **cpu.as_dict(),
        "rdma_witness": witness.as_dict(),
        "fabric_health": {
            d: {
                k: fabric1[d].get(k, 0) - fabric0[d].get(k, 0)
                for k in ("out_of_buffer", "packet_seq_err", "local_ack_timeout_err",
                          "rnr_nak_retry_err", "req_cqe_error", "resp_cqe_error")
                if k in fabric1.get(d, {})
            }
            for d in fabric1
        },
    }
    store.close()
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="B3: ResNet-50 training step time by backend")
    ap.add_argument("--backend", required=True, choices=["http", "rdma", "local"])
    ap.add_argument("--layout", default="raw", choices=["raw", "jpeg"])
    ap.add_argument("--bucket", default="")
    ap.add_argument("--prefix", default="train")
    ap.add_argument("--endpoints", default="aistor1:9000,aistor2:9000")
    ap.add_argument("--servers", default="aistor1:9000,aistor2:9000")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--steps", type=int, default=200, help="measured steps (after warmup)")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--crop", type=int, default=224)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "off"])
    ap.add_argument("--no-channels-last", action="store_true")
    ap.add_argument("--loader-only", action="store_true",
                    help="B2: skip forward/backward and measure how fast the loader "
                         "alone can deliver batches. Answers 'how many GPUs could this "
                         "transport feed?', which end-to-end step time cannot when the "
                         "model is the bottleneck.")
    ap.add_argument("--prefetch", type=int, default=2)
    ap.add_argument("--fetch-workers", type=int, default=4,
                    help="shards fetched concurrently. RDMA transfers a whole object "
                         "per request (no offset in the C ABI), so this is its only "
                         "source of concurrency; HTTP additionally splits each object "
                         "into --http-parts ranges. Report both.")
    ap.add_argument("--max-shards", type=int, default=0)
    ap.add_argument("--epochs-over-shards", type=int, default=4,
                    help="repeat the shard list this many times so long runs do not "
                         "run out of data")
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--http-parts", type=int, default=8)
    ap.add_argument("--http-threads", type=int, default=16)
    ap.add_argument("--local-root", default="")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)

    if not args.bucket:
        args.bucket = "imagenet-raw" if args.layout == "raw" else "imagenet-shards"

    if args.layout == "jpeg" and args.backend == "rdma":
        print("ERROR: --layout jpeg cannot use --backend rdma.\n"
              "  RDMA requires a CUDA device destination; nvJPEG requires the encoded\n"
              "  bytes in host memory. See plans/reports/02-where-rdma-applies.md.\n"
              "  Use --layout raw for the RDMA data path.", file=sys.stderr)
        return 2

    # torchrun sets these; a single-process run does not.
    env_rank = os.environ.get("RANK")
    if env_rank is not None:
        rank = int(env_rank)
        world = int(os.environ["WORLD_SIZE"])
        dist.init_process_group("nccl")
    else:
        rank, world = 0, 1

    result = run(args, rank, world)

    if world > 1:
        gathered = [None] * world
        dist.all_gather_object(gathered, result)
        dist.barrier()
    else:
        gathered = [result]

    if rank == 0:
        agg = _summarise(gathered, args)
        _print(agg, gathered, args)
        rec = RunRecord(
            benchmark="b3-resnet50" + (f"-{args.tag}" if args.tag else ""),
            backend=args.backend,
            params={
                "layout": args.layout, "bucket": args.bucket, "batch_size": args.batch_size,
                "steps": args.steps, "warmup": args.warmup, "gpus": world, "amp": args.amp,
                "crop": args.crop, "prefetch": args.prefetch,
                "fetch_workers": args.fetch_workers, "http_parts": args.http_parts,
                "channels_last": not args.no_channels_last,
                "loader_only": args.loader_only,
                "endpoints": args.endpoints,
            },
            results={"aggregate": agg, "ranks": gathered},
            environment={"gpus": gpu_info(), "torch": torch.__version__},
        )
        print(f"\nsaved: {rec.save(args.results_dir)}")

    if world > 1:
        dist.destroy_process_group()
    return 0


def _summarise(ranks: list[dict], args) -> dict:
    walls = [r["wall_s"] for r in ranks]
    imgs = sum(r["images"] for r in ranks)
    wall = max(walls)
    return {
        "gpus": len(ranks),
        "steps_measured": ranks[0]["steps_measured"],
        "global_batch": args.batch_size * len(ranks),
        "wall_s": wall,
        "seconds_for_n_batches": wall,
        "images_total": imgs,
        "images_per_s": imgs / wall if wall > 0 else None,
        "sec_per_step_median": statistics.median(
            [r["sec_per_step_median"] for r in ranks if r["sec_per_step_median"]]
        ) if any(r["sec_per_step_median"] for r in ranks) else None,
        "fetch_wait_s_max": max(r["window"]["fetch_wait_s"] for r in ranks),
        "fetch_wait_fraction_max": max(
            (r["fetch_wait_fraction"] or 0.0) for r in ranks
        ),
        "fetch_gbps_total": sum(
            (r["window"]["fetch_gbps_effective"] or 0.0) for r in ranks
        ),
        "bytes_fetched_total": sum(r["window"]["bytes_fetched"] for r in ranks),
        "shards_in_window": sum(r["window"]["shards"] for r in ranks),
        "cpu_cores_used": ranks[0]["cpu_cores_used"],
        # MAX, not sum. Every rank scrapes the same CLUSTER-WIDE counter, so each
        # one already sees all ranks' traffic; summing them inflated this figure by
        # the rank count (8.67x at 8 GPUs) and made the proof check below 8x too
        # lenient. The max is one rank's view of the whole window.
        "rdma_read_bytes_total": max(
            (r["rdma_witness"]["rdma_read_bytes"] or 0) for r in ranks
        ),
        "rdma_read_bytes_per_rank": [
            r["rdma_witness"]["rdma_read_bytes"] for r in ranks
        ],
        "fabric_errors_total": sum(
            v for r in ranks for dev in r["fabric_health"].values() for v in dev.values()
        ),
    }


def _print(agg: dict, ranks: list[dict], args) -> None:
    print("\n" + "=" * 78)
    label = "B2 loader-only" if args.loader_only else "B3 ResNet-50"
    print(f"{label} | backend={args.backend} layout={args.layout} "
          f"gpus={agg['gpus']} batch={args.batch_size}/gpu")
    print("=" * 78)
    print(f"  seconds for {agg['steps_measured']} batches : {agg['seconds_for_n_batches']:.2f} s")
    print(f"  images/s                        : {agg['images_per_s']:.0f}")
    print(f"  median step time                : {agg['sec_per_step_median'] * 1e3:.1f} ms")
    print(f"  storage read                    : "
          f"{agg['bytes_fetched_total'] / 2**30:.1f} GiB @ {agg['fetch_gbps_total']:.2f} GB/s")
    print(f"  time blocked on storage         : {agg['fetch_wait_s_max']:.2f} s "
          f"({agg['fetch_wait_fraction_max'] * 100:.1f}% of wall)")
    print(f"  host CPU                        : {agg['cpu_cores_used']:.1f} cores")
    print(f"  server RDMA bytes               : {agg['rdma_read_bytes_total']:,}")
    print(f"  fabric errors                   : {agg['fabric_errors_total']}")
    r0 = ranks[0]["window"]
    print(f"  shards read in window           : {agg['shards_in_window']}")
    print(f"  rank0 breakdown                 : fetch_wait {r0['fetch_wait_s']:.2f}s | "
          f"decode {r0['decode_s']:.2f}s | augment {r0['augment_s']:.2f}s | "
          f"compute {ranks[0]['compute_s_total']:.2f}s")
    if args.backend == "rdma" and agg["rdma_read_bytes_total"] < agg["bytes_fetched_total"] * 0.5:
        print("\n  WARNING: server RDMA counters did not account for the bytes read.")
        print("           This run silently used HTTP; do not report it as RDMA.")


if __name__ == "__main__":
    sys.exit(main())
