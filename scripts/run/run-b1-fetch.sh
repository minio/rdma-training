#!/usr/bin/env bash
# Copyright 2026 MinIO, Inc.
# SPDX-License-Identifier: Apache-2.0

# B1 — raw object GET throughput into GPU memory, for every backend.
#
# Runs the same size x concurrency matrix against each transport so the cells are
# directly comparable. Objects are created once (over HTTP) and reused, so both
# transports read byte-identical objects from the same drives.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
# shellcheck disable=SC1091
source ./env.sh

ENDPOINTS="${ENDPOINTS:-aistor1:9000,aistor2:9000}"
SIZES="${SIZES:-1,4,16,64,256}"
CONCURRENCY="${CONCURRENCY:-1,4,8,16,32,64}"
ITERS="${ITERS:-6}"
OBJECTS="${OBJECTS:-64}"
TAG="${TAG:-dual-node}"
BACKENDS="${BACKENDS:-rdma http}"
LOGDIR="${LOGDIR:-$REPO_ROOT/results/logs}"

mkdir -p "$LOGDIR"

for BE in $BACKENDS; do
    log="$LOGDIR/b1-$BE-$TAG.log"
    echo "== B1 $BE (log: $log) =="
    if python -u -m s3rdma_train.bench_fetch \
        --backend "$BE" \
        --endpoints "$ENDPOINTS" \
        --sizes "$SIZES" \
        --concurrency "$CONCURRENCY" \
        --iters "$ITERS" \
        --objects "$OBJECTS" \
        --tag "$TAG" >"$log" 2>&1; then
        grep -vE "FutureWarning|import pynvml" "$log" | tail -45
    else
        echo "FAILED (exit $?) -- tail of log:"
        grep -vE "FutureWarning|import pynvml" "$log" | tail -30
    fi
    echo
done

echo "== done. raw JSON in results/ =="
