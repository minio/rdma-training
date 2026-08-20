#!/usr/bin/env bash
# Copyright 2026 MinIO, Inc.
# SPDX-License-Identifier: Apache-2.0

# B4 fairness check: a DDP-sharded checkpoint.
#
# The single-process checkpoint numbers are per-process, and the Python HTTP client
# is capped near 1.4 GB/s per process for writes (report 01 §3). A real 8-GPU job
# does not write its checkpoint from one process -- each rank writes its own shard.
# That gives HTTP roughly 8x more headroom and is the configuration it should be
# judged in, so we measure it rather than argue about it.
#
# Each of N ranks saves TOTAL_GIB/N from its own GPU, concurrently.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
# shellcheck disable=SC1091
source ./env.sh
ulimit -n 65536

RANKS="${RANKS:-8}"
TOTAL_GIB="${TOTAL_GIB:-32}"
REPEATS="${REPEATS:-3}"
CHUNK_MIB="${CHUNK_MIB:-512}"
CONCURRENCY="${CONCURRENCY:-8}"
LOGDIR="$REPO_ROOT/results/logs"
mkdir -p "$LOGDIR"

PER_RANK=$(python3 -c "print($TOTAL_GIB / $RANKS)")
echo "== DDP-sharded checkpoint: $RANKS ranks x ${PER_RANK} GiB = ${TOTAL_GIB} GiB total =="
echo "   chunk=${CHUNK_MIB} MiB concurrency=${CONCURRENCY} repeats=${REPEATS}"

cpu_snapshot() {
    awk '/^cpu /{idle=$5+$6; total=0; for(i=2;i<=NF;i++) total+=$i; print total-idle, total}' /proc/stat
}

# Order matters for a live demo: BACKENDS="http rdma" shows the baseline first and
# the improvement second.
for BE in ${BACKENDS:-rdma http}; do
    echo
    echo "-- backend=$BE --"
    # Clear this backend's per-rank logs first. The aggregate below is computed by
    # scanning them, so logs left over from a previous run with MORE ranks would be
    # counted too -- silently inflating the result.
    rm -f "$LOGDIR/b4ddp-$BE-rank"*.log

    before=$(cpu_snapshot)
    start=$(python3 -c 'import time; print(time.time())')

    pids=()
    for r in $(seq 0 $((RANKS - 1))); do
        python -u -m s3rdma_train.bench_ckpt \
            --backends "$BE" --methods flat \
            --sizes-gib "$PER_RANK" --repeats "$REPEATS" \
            --chunk-mib "$CHUNK_MIB" --concurrency "$CONCURRENCY" \
            --device "cuda:$r" --skip-load \
            --bucket checkpoints --tag "ddp-r$r-$BE" \
            >"$LOGDIR/b4ddp-$BE-rank$r.log" 2>&1 &
        pids+=($!)
    done
    failed=0
    for p in "${pids[@]}"; do wait "$p" || failed=$((failed + 1)); done

    end=$(python3 -c 'import time; print(time.time())')
    after=$(cpu_snapshot)

    python3 - "$start" "$end" "$before" "$after" "$RANKS" "$TOTAL_GIB" "$failed" <<'PY'
import os, sys
start, end = float(sys.argv[1]), float(sys.argv[2])
b0, t0 = map(float, sys.argv[3].split())
b1, t1 = map(float, sys.argv[4].split())
ranks, total_gib, failed = int(sys.argv[5]), float(sys.argv[6]), int(sys.argv[7])
wall = end - start
cores = (b1 - b0) / (t1 - t0) * (os.cpu_count() or 1) if t1 > t0 else 0.0
print(f"   ranks failed                          : {failed}")
print(f"   aggregate wall (incl. process startup): {wall:.2f} s")
print(f"   whole-host CPU during window          : {cores:.1f} cores")
PY

    echo "   per-rank save (median of $REPEATS repeats, excludes startup):"
    grep -h "synthetic" "$LOGDIR/b4ddp-$BE-rank"*.log 2>/dev/null |
        awk '{printf "     %s\n", $0}' | head -"$RANKS"

    # Aggregate steady-state rate: per-rank GB/s summed. This is the figure to
    # compare against the single-process numbers, since it excludes the fixed
    # cost of spawning 8 interpreters and 8 CUDA contexts.
    grep -h "synthetic" "$LOGDIR/b4ddp-$BE-rank"*.log 2>/dev/null |
        awk '{s+=$6; n++} END {if (n) printf "   aggregate steady save rate            : %.2f GB/s across %d ranks\n", s, n}'
done

echo
echo "== done =="
