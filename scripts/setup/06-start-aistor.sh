#!/usr/bin/env bash
# Copyright 2026 MinIO, Inc.
# SPDX-License-Identifier: Apache-2.0

# Install the systemd unit + environment for the AIStor RDMA cluster and start it.
# Run on EVERY storage node (the nodes discover each other from MINIO_VOLUMES).
#
# Topology: one server pool spanning both nodes, 24 drives each = 48 drives.
#   erasure set size 16 -> 3 sets, EC:4 (12 data + 4 parity)
#
# Set MINIO_LICENSE in the environment before running, or place the license at
# /etc/minio/minio.license.
set -euo pipefail

BIN=/usr/local/bin/minio.rdma

# NOTE: keep this default in its own single-quoted assignment. Inlining it as
# ${MINIO_VOLUMES:-http://aistor{1...2}/...} silently corrupts the pattern --
# shell parameter expansion ends at the first unescaped `}`, so the brace after
# `2` closes the expansion and the rest is appended as literal text, yielding
# `http://aistor{1...2:9000/...}`. MinIO then rejects it with a misleading
# "allowed minimum range of 4" message (there is no such minimum; a 2-host
# ellipsis is valid).
DEFAULT_VOLUMES='http://aistor{1...2}:9000/mnt/data{1...24}/minio'
VOLUMES="${MINIO_VOLUMES:-$DEFAULT_VOLUMES}"
ROOT_USER="${MINIO_ROOT_USER:-minioadmin}"
ROOT_PASSWORD="${MINIO_ROOT_PASSWORD:-minioadmin}"
LICENSE_FILE=/etc/minio/minio.license

[[ -x "$BIN" ]] || { echo "$BIN not found; run 05-install-aistor.sh first" >&2; exit 1; }

echo "== configuring aistor on $(hostname) =="

sudo mkdir -p /etc/minio

if [[ -n "${MINIO_LICENSE:-}" ]]; then
    printf '%s' "$MINIO_LICENSE" | sudo tee "$LICENSE_FILE" >/dev/null
    sudo chmod 600 "$LICENSE_FILE"
fi
[[ -s "$LICENSE_FILE" ]] || { echo "no license at $LICENSE_FILE and MINIO_LICENSE unset" >&2; exit 1; }

sudo tee /etc/minio/config.env >/dev/null <<EOF
# ---- managed by ai-training/scripts/setup/06-start-aistor.sh ----

MINIO_VOLUMES="$VOLUMES"
MINIO_ROOT_USER="$ROOT_USER"
MINIO_ROOT_PASSWORD="$ROOT_PASSWORD"

# The license MUST be passed as the --license flag. That CLI flag has no EnvVar
# binding in the server (unlike most MinIO flags), so setting MINIO_LICENSE or
# MINIO_LICENSE_FILE in the environment is silently ignored and the server comes
# up in offline mode with "All S3 operations are denied".
MINIO_OPTS="--address :9000 --console-address :9001 --license $LICENSE_FILE"

# Erasure layout: 48 drives / set size 16 = 3 sets, EC:4 (12 data + 4 parity).
# Reads need only 12 of 16 shards, so the read path (the training data path) is
# wide; writes pay 16/12 = 1.33x amplification, which is what production looks
# like and is therefore what the checkpoint benchmark should measure.
MINIO_ERASURE_SET_DRIVE_COUNT=16

# ---- RDMA ----
# Inter-node RDMA: erasure shard traffic between coe02 and coe04 moves over RoCE
# instead of TCP. Independent of the client-facing GPU-Direct path, which is
# always available on this build and is negotiated per request via the
# x-amz-rdma-token header.
MINIO_RDMA_INTERNODE=on

# 400 GbE tuning (from aistor docs "High-Throughput Configuration").
MINIO_RDMA_CHANNELS=256
MINIO_RDMA_NUM_DCIS=256
MINIO_RDMA_CQ_DEPTH=8192
MINIO_RDMA_DELAY_INTERVAL=500

# Leave the adaptive inflight-bytes window on AUTO. It is the primary incast
# control and it self-tunes; pinning it is only useful when characterising a
# known-lossy fabric.
MINIO_RDMA_INTERNODE_INFLIGHT_BYTES=0

# Traffic class AUTO: the library reads the host DCB config (which
# 04-roce-lossless.sh set to PFC prio 3 / DSCP 26) and marks its QPs to land on
# that lossless class. Do not hardcode unless the fabric trusts PCP instead.
MINIO_RDMA_INTERNODE_TRAFFIC_CLASS=0

# ---- observability ----
# Public metrics so the benchmark harness can scrape RDMA byte counters without
# minting a token per run. Lab-only setting.
MINIO_PROMETHEUS_AUTH_TYPE=public
EOF

sudo tee /etc/systemd/system/minio.service >/dev/null <<EOF
[Unit]
Description=MinIO AIStor (GPU-Direct RDMA)
Documentation=https://docs.min.io
Wants=network-online.target
After=network-online.target
AssertFileIsExecutable=$BIN

[Service]
Type=notify
WorkingDirectory=/usr/local
User=root
Group=root
EnvironmentFile=/etc/minio/config.env
ExecStart=$BIN server \$MINIO_OPTS \$MINIO_VOLUMES
Restart=always
RestartSec=5

# RDMA buffer registration needs unlimited locked memory.
LimitMEMLOCK=infinity
LimitNOFILE=1048576

MemoryAccounting=no
TasksMax=infinity
TimeoutSec=infinity
OOMScoreAdjust=-1000
SendSIGKILL=no

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable minio >/dev/null
sudo systemctl restart minio

echo "-- waiting for service --"
for _ in $(seq 1 40); do
    if curl -sf --max-time 2 "http://127.0.0.1:9000/minio/health/live" >/dev/null 2>&1; then
        echo "healthy"
        break
    fi
    sleep 2
done

systemctl is-active minio || true
echo
echo "-- recent log --"
sudo journalctl -u minio -n 25 --no-pager | tail -25
