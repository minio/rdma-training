#!/usr/bin/env bash
# Copyright 2026 MinIO, Inc.
# SPDX-License-Identifier: Apache-2.0

# Stop any memkv / minio instances on this storage node so its NVMe drives can be
# reformatted for AIStor. Idempotent: safe to run when nothing is running.
set -euo pipefail

echo "== stopping memkv/minio on $(hostname) =="

for unit in minio minio-pool1 minio-pool2 memkv; do
    if systemctl is-active --quiet "$unit" 2>/dev/null; then
        echo "stopping systemd unit: $unit"
        sudo systemctl stop "$unit"
    fi
done

# memkv here is started via `sudo nohup memkv start ...`, not systemd.
if pgrep -f 'memkv start' >/dev/null 2>&1; then
    echo "killing memkv processes"
    sudo pkill -f 'memkv start' || true
    for _ in $(seq 1 30); do
        pgrep -f 'memkv start' >/dev/null 2>&1 || break
        sleep 1
    done
    if pgrep -f 'memkv start' >/dev/null 2>&1; then
        echo "memkv did not exit; sending SIGKILL"
        sudo pkill -9 -f 'memkv start' || true
        sleep 2
    fi
fi

if pgrep -x minio >/dev/null 2>&1; then
    echo "killing minio processes"
    sudo pkill -x minio || true
    sleep 3
fi

echo "-- remaining matches (should be empty) --"
pgrep -af 'memkv start|^minio' || echo "(none)"
echo "== done =="
