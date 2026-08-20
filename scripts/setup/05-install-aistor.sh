#!/usr/bin/env bash
# Copyright 2026 MinIO, Inc.
# SPDX-License-Identifier: Apache-2.0

# Install the latest MinIO AIStor **RDMA** release on this storage node.
#
# The plain `minio` build cannot do GPU-Direct RDMA — the feature needs a CGO
# build (tags `rdma,kqueue`) linked against libibverbs/librdmacm and NVIDIA's
# cuObject libraries. That build ships as a separate `minio.rdma` artifact whose
# package also installs the required shared objects into /usr/lib/minio:
#
#   libs3rdma.so       - S3-over-RDMA (client-facing GPU-Direct) transport
#   libp2p_rdma.so     - inter-node RDMA (erasure shard transfers)
#   libminiocpp.so     - minio-cpp with RDMA
#   libcuobjclient.so  - NVIDIA cuObject client
#   libcufile.so       - NVIDIA GPUDirect Storage
#
# Verify with `minio --version`; it must report "Features: GPU-Direct RDMA".
set -euo pipefail

BASE="${BASE:-https://dl.min.io/aistor/minio/release/linux-amd64}"
PKG="${PKG:-minio.rdma_20260807183435.0.0_amd64.deb}"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo "== installing $PKG on $(hostname) =="

curl -fsSL --retry 3 -o "$TMP/$PKG" "$BASE/$PKG"
curl -fsSL --retry 3 -o "$TMP/$PKG.sha256sum" "$BASE/$PKG.sha256sum" || true

if [[ -s "$TMP/$PKG.sha256sum" ]]; then
    # The published sha256sum names the file by its full path; compare digests only.
    want=$(awk '{print $1}' "$TMP/$PKG.sha256sum")
    got=$(sha256sum "$TMP/$PKG" | awk '{print $1}')
    if [[ "$want" != "$got" ]]; then
        echo "checksum mismatch: want $want got $got" >&2
        exit 1
    fi
    echo "checksum ok: $got"
fi

# The stock `minio` package installs the same /usr/local/bin/minio path, so remove
# it first rather than letting dpkg fail on the file conflict.
if dpkg -l minio 2>/dev/null | grep -q '^ii'; then
    echo "removing conflicting 'minio' package"
    sudo dpkg -r --force-depends minio || true
fi

# Only reach for apt if a dependency is actually missing. On a rig where the RDMA
# stack is already installed (OFED hosts normally have all three) this avoids
# touching the package database at all, which matters if apt is wedged for an
# unrelated reason.
missing=()
for p in libibverbs1 librdmacm1 libnuma1; do
    dpkg -l "$p" 2>/dev/null | grep -q '^ii' || missing+=("$p")
done
if [[ ${#missing[@]} -gt 0 ]]; then
    echo "installing missing deps: ${missing[*]}"
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${missing[@]}" >/dev/null
else
    echo "deps already satisfied: libibverbs1 librdmacm1 libnuma1"
fi

sudo dpkg -i "$TMP/$PKG"

echo
echo "== verification =="
# The RDMA flavour installs as `minio.rdma`, deliberately distinct from the stock
# `minio` binary so both can coexist. Everything downstream uses this path.
BIN=/usr/local/bin/minio.rdma
if [[ ! -x "$BIN" ]]; then
    echo "ERROR: $BIN not found after install" >&2
    dpkg -L minio.rdma | grep bin/ >&2 || true
    exit 1
fi
echo "binary: $BIN"
"$BIN" --version
echo
echo "-- RDMA shared objects --"
ls -1 /usr/lib/minio/*.so* 2>/dev/null || echo "  MISSING /usr/lib/minio/*.so"
echo
echo "-- dynamic linkage --"
ldd "$BIN" 2>/dev/null | grep -iE "ibverbs|rdmacm|numa|cuobj|cufile|s3rdma|p2p_rdma" ||
    echo "  (statically linked or dlopened at runtime)"

# Match with bash's pattern test rather than `... | grep -q`: under pipefail,
# grep -q closing the pipe early makes the producer take SIGPIPE and the whole
# pipeline report failure, inverting the result of a successful match.
version_out=$("$BIN" --version 2>&1 || true)
if [[ "$version_out" != *"GPU-Direct RDMA"* ]]; then
    echo
    echo "ERROR: binary does not report 'Features: GPU-Direct RDMA'" >&2
    exit 1
fi
echo
echo "== ok: GPU-Direct RDMA build installed =="
