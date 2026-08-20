#!/usr/bin/env bash
# Copyright 2026 MinIO, Inc.
# SPDX-License-Identifier: Apache-2.0

# B2 + B3 — the training benchmarks, for every valid (layout, backend) pair.
#
# B3 is the headline: seconds for N batches of ResNet-50. B2 runs the identical
# pipeline with the model removed, which answers the question B3 cannot when the
# model is the bottleneck: how many GPUs could this transport actually feed?
#
# Both run at 8 GPUs via torchrun, so each rank is its own process. That matters for
# the HTTP arm -- a single CPython process caps at ~2 GB/s regardless of threads, and
# worker processes are how real DataLoaders get past it.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
# shellcheck disable=SC1091
source ./env.sh
ulimit -n 65536

GPUS="${GPUS:-8}"
STEPS="${STEPS:-200}"
WARMUP="${WARMUP:-20}"
BATCH="${BATCH:-256}"
FETCH_WORKERS="${FETCH_WORKERS:-8}"
# B2 needs far more steps than B3: with no model to slow it down, 200 batches take a
# fraction of a second, and a window that short mostly measures already-prefetched
# buffers rather than the transport.
B2_STEPS="${B2_STEPS:-3000}"
TAG="${TAG:-ddp$GPUS}"
LOGDIR="$REPO_ROOT/results/logs"
mkdir -p "$LOGDIR"

run_one() {
    local mode="$1" backend="$2" layout="$3" extra="${4:-}"
    local steps="$STEPS"
    [ "$mode" = b2 ] && steps="$B2_STEPS"
    local name="$mode-$backend-$layout-$TAG"
    local log="$LOGDIR/$name.log"
    echo "== $mode | backend=$backend layout=$layout gpus=$GPUS =="
    # shellcheck disable=SC2086
    if torchrun --standalone --nproc-per-node="$GPUS" \
        -m s3rdma_train.train_resnet50 \
        --backend "$backend" --layout "$layout" \
        --steps "$steps" --warmup "$WARMUP" --batch-size "$BATCH" \
        --fetch-workers "$FETCH_WORKERS" \
        --tag "$mode-$TAG" $extra >"$log" 2>&1; then
        grep -vE "FutureWarning|import pynvml|CUDACachingAllocator" "$log" | tail -18
    else
        echo "   FAILED (exit $?):"
        grep -vE "FutureWarning|import pynvml" "$log" | tail -20
    fi
    echo
}

# B3: end-to-end ResNet-50. raw layout for both transports (the fair comparison),
# then the JPEG layout over HTTP to show where a conventional pipeline's time goes.
run_one b3 rdma raw
run_one b3 http raw
run_one b3 http jpeg

# B2: same pipeline, no model. The loader's ceiling.
run_one b2 rdma raw "--loader-only"
run_one b2 http raw "--loader-only"

echo "== done. raw JSON in results/ =="
