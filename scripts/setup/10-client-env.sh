#!/usr/bin/env bash
# Copyright 2026 MinIO, Inc.
# SPDX-License-Identifier: Apache-2.0

# Set up the training client (the GPU node): PyTorch, and the RDMA client
# libraries needed for GPU-Direct S3.
#
# The client needs three shared objects that the server-side `minio.rdma` package
# does NOT ship (the server links libminiocpp at build time but never calls it,
# so the release package omits it):
#
#   libminiocpp.so.0.4.0     minio-cpp built with -DMINIO_CPP_ENABLE_RDMA=ON;
#                            exposes the stable C ABI this project binds to
#   libcuobjclient.so.1.2.0  NVIDIA cuObject client
#   libcufile.so.1.18.0      NVIDIA GPUDirect Storage
#
# Source them, in order of preference:
#   1. --libs-from <dir>   an explicit directory
#   2. $RDMA_LIB_DIR       same, from the environment
#   3. build minio-cpp at tag v0.6.0 or later yourself, with
#      -DMINIO_CPP_ENABLE_RDMA=ON (see docs/01-setup-aistor-rdma.md)
#
# Usage: ./10-client-env.sh [--torch-cuda cu130] [--libs-from DIR]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENDOR="$REPO_ROOT/vendor/rdma-libs"
VENV="$REPO_ROOT/.venv"

LIBS_FROM="${LIBS_FROM:-}"
RDMA_LIB_DIR="${RDMA_LIB_DIR:-}"
TORCH_CUDA="${TORCH_CUDA:-cu130}"
TORCH_VERSION="${TORCH_VERSION:-2.13.0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --torch-cuda) TORCH_CUDA="$2"; shift 2 ;;
        --torch-version) TORCH_VERSION="$2"; shift 2 ;;
        --libs-from) LIBS_FROM="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

echo "== client env setup on $(hostname) =="
echo "   repo:   $REPO_ROOT"
echo "   torch:  $TORCH_VERSION ($TORCH_CUDA)"

# ---------------------------------------------------------------- RDMA libs ---
REQUIRED_LIBS=(libminiocpp.so libcuobjclient.so.1 libcufile.so.0)

have_all_libs() {
    local d="$1" l
    for l in "${REQUIRED_LIBS[@]}"; do
        [[ -e "$d/$l" ]] || return 1
    done
    return 0
}

if ! have_all_libs "$VENDOR"; then
    echo
    echo "-- staging RDMA client libraries --"
    mkdir -p "$VENDOR"

    src=""
    if [[ -n "$LIBS_FROM" ]]; then
        src="$LIBS_FROM"
    else
        for cand in \
            "${RDMA_LIB_DIR:-}" \
            "$REPO_ROOT/vendor/rdma-libs"; do
            [[ -n "$cand" && -e "$cand/libminiocpp.so" ]] && { src="$cand"; break; }
        done
    fi

    if [[ -z "$src" || ! -e "$src/libminiocpp.so" ]]; then
        cat >&2 <<'MSG'
ERROR: could not find the RDMA client libraries.

Provide them with one of:
  --libs-from <dir>     directory containing libminiocpp.so, libcuobjclient.so.1,
                        libcufile.so.0 (plus their versioned targets)
  RDMA_LIB_DIR=<dir>    the same directory, via the environment

Or build minio-cpp yourself (v0.6.0 is the first release with every RDMA fix
this project needs -- see docs/01-setup-aistor-rdma.md):
  git clone --branch v0.6.0 https://github.com/minio/minio-cpp
  cmake . -B build -DBUILD_SHARED_LIBS=ON -DMINIO_CPP_ENABLE_RDMA:BOOL=ON
  cmake --build build -j
MSG
        exit 1
    fi

    echo "   source: $src"
    # -a preserves the SONAME symlinks, which libminiocpp's $ORIGIN rpath relies
    # on to find libcuobjclient/libcufile beside itself.
    cp -a "$src"/libminiocpp.so* "$VENDOR"/
    cp -a "$src"/libcuobjclient.so* "$VENDOR"/
    cp -a "$src"/libcufile.so* "$VENDOR"/
    # Server-side library: not needed on a client, and keeping it invites
    # confusion about which side of the transfer this host is on.
    rm -f "$VENDOR"/libcuobjserver.so*
    echo "   -> $VENDOR"
else
    echo "-- RDMA client libraries already present in $VENDOR --"
fi

ls -1 "$VENDOR" | sed 's/^/     /'

# Match with bash's own pattern test, NOT `... | grep -q`. Under `set -o
# pipefail`, grep -q exits on the first match and closes the pipe, the producer
# dies with SIGPIPE (141), and the pipeline reports failure -- so a SUCCESSFUL
# match reads as "symbol missing". No pipe, no problem.
abi_syms=$(nm -D "$VENDOR/libminiocpp.so" 2>/dev/null || true)
if [[ "$abi_syms" != *" T miniocpp_get_object"* ]]; then
    echo "ERROR: $VENDOR/libminiocpp.so does not export the miniocpp_* C ABI." >&2
    echo "       An older minio-cpp build (pre-v0.4.0) will not work." >&2
    exit 1
fi
# grep -c consumes all input, so it cannot trigger the SIGPIPE case above.
echo "   C ABI ok: $(grep -c ' T miniocpp_' <<<"$abi_syms") symbols"

# Which libcufile ends up in the process decides whether RDMA works at all, so
# resolve it here rather than leaving it to load order.
CUFILE_REAL=$(ls "$VENDOR"/libcufile.so.1.* 2>/dev/null | sort -V | tail -1)
[[ -n "$CUFILE_REAL" ]] || { echo "no versioned libcufile in $VENDOR" >&2; exit 1; }


# ------------------------------------------------------------------- python ---
echo
echo "-- python environment --"
command -v uv >/dev/null || { echo "uv not found; install from https://astral.sh/uv" >&2; exit 1; }

[[ -d "$VENV" ]] || uv venv --python 3.12 "$VENV"

# torch/torchvision come from the CUDA-specific index; everything else from PyPI.
uv pip install --python "$VENV/bin/python" -q \
    --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" \
    --extra-index-url https://pypi.org/simple \
    "torch==${TORCH_VERSION}" torchvision

uv pip install --python "$VENV/bin/python" -q \
    numpy pandas boto3 minio nvidia-ml-py psutil requests tabulate \
    safetensors huggingface_hub

echo
echo "-- versions --"
LD_PRELOAD="$CUFILE_REAL" MINIOCPP_LIB="$VENDOR/libminiocpp.so" \
    LD_LIBRARY_PATH="$VENDOR:${LD_LIBRARY_PATH:-}" \
    PYTHONPATH="$REPO_ROOT/src" "$VENV/bin/python" - <<'PY'
import torch, torchvision
print(f"  torch        {torch.__version__}")
print(f"  torchvision  {torchvision.__version__}")
print(f"  cuda avail   {torch.cuda.is_available()}  devices={torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"  cuda runtime {torch.version.cuda}")
    print(f"  device 0     {torch.cuda.get_device_name(0)}")

# Deliberately after torch: that is the load order that exposes the libcufile
# SONAME collision, so it is the order worth asserting.
from s3rdma_train import rdma_client
print(f"  libminiocpp  {'loaded' if rdma_client.is_available() else 'FAILED'}")
print(f"  rdma ready   {rdma_client.is_available()}")
PY

cat > "$REPO_ROOT/env.sh" <<EOF
# source this before running any benchmark
export S3RDMA_ROOT="$REPO_ROOT"
export MINIOCPP_LIB="$VENDOR/libminiocpp.so"
export LD_LIBRARY_PATH="$VENDOR:\${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_ROOT/src:\${PYTHONPATH:-}"
export PATH="$VENV/bin:\$PATH"

# ---------------------------------------------------------------------------
# LD_PRELOAD is REQUIRED, not an optimisation.
#
# PyTorch's CUDA wheels bundle their own libcufile (nvidia/cu13/lib,
# cuFile 1.15.1). libcuobjclient needs cuFileRDMADescStrGet, which only exists
# in cuFile >= 1.18. Both files carry SONAME libcufile.so.0, so whichever is
# dlopen'd first claims that name and the other is ignored -- dlopen by absolute
# path will NOT override an already-loaded SONAME.
#
# Import torch first without this and libminiocpp fails to load with:
#   libcuobjclient.so.1: undefined symbol: _Z20cuFileRDMADescStrGet...
#
# Preloading 1.18.0 makes it win for both consumers; torch is happy with it.
# ---------------------------------------------------------------------------
export LD_PRELOAD="$CUFILE_REAL\${LD_PRELOAD:+:\$LD_PRELOAD}"
EOF

echo
echo "== done. run:  source $REPO_ROOT/env.sh =="
