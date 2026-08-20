# Copyright 2026 MinIO, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Thin ctypes binding to libminiocpp's stable C ABI (GPU-Direct S3 over RDMA).

This talks directly to ``libminiocpp.so`` (minio-cpp >= v0.4.0, built with
``-DMINIO_CPP_ENABLE_RDMA=ON``), which is the same library that ships inside the
``minio.rdma`` server package. Binding here rather than depending on an
SDK branch keeps this project self-contained: the only external artifact needed
is the ``.so`` the storage nodes already install.

The whole GPU-Direct integration is one fact: ``miniocpp_get_object`` /
``miniocpp_put_object`` take a plain ``void*``, and cuObject registers it whether
it is host or device memory. So a CUDA tensor's ``data_ptr()`` can be handed
across unchanged and the NIC DMAs straight into VRAM.

    buf = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    n = client.get("bucket", "key", buf.data_ptr(), nbytes)

IMPORTANT — silent HTTP fallback
--------------------------------
Per the C ABI contract, when ``buf`` is non-NULL the library "attempts RDMA with
HTTP fallback". A successful return therefore does **not** prove RDMA was used;
the transfer may have quietly gone over TCP. Any benchmark that compares RDMA
against HTTP must confirm RDMA independently, by reading the server's
``minio_api_rdma_read_bytes_total`` / ``minio_api_rdma_write_bytes_total``
counters across the run. See ``metrics.RDMAWitness``.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
from typing import Optional, Union

__all__ = [
    "RDMANotAvailable",
    "RDMAError",
    "RDMADeclined",
    "RDMAClient",
    "is_available",
    "library_path",
]

# Error codes from c_api.h. >= 0 is the transferred byte count.
_ERR_GENERIC = -1
_ERR_RDMA_DECLINED = -2
_ERR_INVALID_ARG = -3

# A single cuObject registration (cuMemObjGetDescriptor) can pin at most 4 GiB;
# minio-cpp names this kCuObjMaxMemoryRegSize. Its source says RDMA is simply not
# attempted above the limit and the transfer falls back to one ordinary HTTP PUT,
# but measured behaviour is worse than that: 3.5 GiB transfers fine and 4 GiB
# **segfaults** inside miniocpp_put_object. We refuse the call rather than let the
# process die, and callers should chunk below this (see checkpoint.py).
MAX_REGISTRATION_BYTES = 4 * 1024 * 1024 * 1024

# Stay a margin under the hard limit: the registration covers the whole buffer, and
# a request exactly at the boundary is the one observed to crash.
SAFE_TRANSFER_BYTES = 2 * 1024 * 1024 * 1024

BufferLike = Union[int, memoryview, bytes, bytearray]


class RDMANotAvailable(RuntimeError):
    """libminiocpp.so could not be loaded."""


class RDMAError(RuntimeError):
    """libminiocpp.so returned an error."""


class RDMADeclined(RDMAError):
    """The server explicitly declined RDMA for this request."""


_lib: Optional[ctypes.CDLL] = None
_lib_path: Optional[str] = None
_has_range: bool = False


def library_path() -> Optional[str]:
    """Path of the loaded libminiocpp.so, or None if not loaded yet."""
    return _lib_path


def has_range() -> bool:
    """True if this libminiocpp exposes miniocpp_get_object_range.

    Without it, objects larger than MAX_REGISTRATION_BYTES cannot be fetched over
    RDMA at all -- and standard Hugging Face safetensors shards are ~5 GB, i.e.
    above it. See minio-cpp PR #258.
    """
    try:
        _load()
    except RDMANotAvailable:
        return False
    return _has_range

def _loaded_cufile() -> Optional[str]:
    """Path of the libcufile already mapped into this process, if any."""
    try:
        with open(f"/proc/{os.getpid()}/maps") as fh:
            for line in fh:
                if "libcufile.so" in line:
                    return line.rsplit(" ", 1)[-1].strip()
    except OSError:
        pass
    return None


def _load_error_help(path: str, exc: Exception) -> str:
    """Turn a dlopen failure into something the reader can act on.

    The overwhelmingly common failure is the libcufile SONAME collision with
    PyTorch's bundled copy, which produces an opaque undefined-symbol error
    naming a mangled C++ symbol. Recognise it and say what to do.
    """
    msg = str(exc)
    lines = [f"failed to load libminiocpp.so ({path}): {msg}"]

    if "cuFileRDMADescStrGet" in msg or ("libcuobjclient" in msg and "undefined symbol" in msg):
        loaded = _loaded_cufile()
        lines += [
            "",
            "CAUSE: the wrong libcufile is loaded in this process.",
            "  libcuobjclient needs cuFileRDMADescStrGet, which exists only in",
            "  cuFile >= 1.18. PyTorch's CUDA wheels bundle cuFile 1.15.x under",
            "  site-packages/nvidia/*/lib. Both carry SONAME libcufile.so.0, so the",
            "  first one loaded wins and dlopen by absolute path will not override it.",
        ]
        if loaded:
            lines.append(f"  already loaded in this process: {loaded}")
        lines += [
            "",
            "FIX: preload the 1.18+ libcufile so it claims the SONAME first:",
            "  export LD_PRELOAD=<repo>/vendor/rdma-libs/libcufile.so.1.18.0",
            "  (sourcing the repo's env.sh does this for you)",
        ]
    else:
        lines += [
            "",
            "Set MINIOCPP_LIB to a libminiocpp.so built from minio-cpp >= v0.4.0 with",
            "-DMINIO_CPP_ENABLE_RDMA=ON, and make sure libcuobjclient.so.1 and",
            "libcufile.so.0 sit beside it (scripts/setup/10-client-env.sh does this).",
        ]
    return "\n".join(lines)


def _load() -> ctypes.CDLL:
    global _lib, _lib_path
    if _lib is not None:
        return _lib

    path = os.environ.get("MINIOCPP_LIB") or ctypes.util.find_library("miniocpp") or "libminiocpp.so"
    try:
        lib = ctypes.CDLL(path)
    except OSError as exc:
        raise RDMANotAvailable(_load_error_help(path, exc)) from exc

    void_p, c_char_p, c_size_t = ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t
    c_ssize_t, c_int = ctypes.c_ssize_t, ctypes.c_int
    char64 = ctypes.c_char * 64

    lib.miniocpp_client_new.argtypes = [c_char_p, c_char_p, c_char_p, c_char_p, c_char_p, c_int]
    lib.miniocpp_client_new.restype = void_p

    lib.miniocpp_client_free.argtypes = [void_p]
    lib.miniocpp_client_free.restype = None

    # read_cb / write_cb are passed as NULL: this binding only uses the
    # direct-buffer (RDMA-capable) path, never callback streaming.
    lib.miniocpp_put_object.argtypes = [
        void_p, c_char_p, c_char_p, void_p, c_size_t, void_p, void_p, char64, char64
    ]
    lib.miniocpp_put_object.restype = c_ssize_t

    lib.miniocpp_get_object.argtypes = [void_p, c_char_p, c_char_p, void_p, c_size_t, void_p, void_p]
    lib.miniocpp_get_object.restype = c_ssize_t

    # Ranged GET. Only present in libminiocpp builds that carry
    # https://github.com/minio/minio-cpp/pull/258; older libraries can only fetch
    # whole objects, which means an object above the 4 GiB registration limit
    # cannot use RDMA at all. has_range() reports which kind you have.
    global _has_range
    _has_range = hasattr(lib, "miniocpp_get_object_range")
    if _has_range:
        lib.miniocpp_get_object_range.argtypes = [
            void_p, c_char_p, c_char_p, void_p, c_size_t, ctypes.c_uint64
        ]
        lib.miniocpp_get_object_range.restype = c_ssize_t

    lib.miniocpp_alloc_aligned.argtypes = [c_size_t]
    lib.miniocpp_alloc_aligned.restype = void_p

    lib.miniocpp_free_aligned.argtypes = [void_p]
    lib.miniocpp_free_aligned.restype = None

    lib.miniocpp_rdma_available.argtypes = []
    lib.miniocpp_rdma_available.restype = c_int

    lib.miniocpp_last_error.argtypes = []
    lib.miniocpp_last_error.restype = c_char_p

    _lib, _lib_path = lib, path
    return lib


def is_available() -> bool:
    """True if the process-wide cuObjClient is connected to a cuObjServer.

    A True here means an RDMA transfer is *likely* to succeed. It is not a
    guarantee for any individual request, and it says nothing about whether a
    completed request actually used RDMA.
    """
    try:
        return _load().miniocpp_rdma_available() != 0
    except RDMANotAvailable:
        return False


def alloc_aligned(size: int) -> int:
    """Page-aligned host buffer suitable for RDMA registration (raw pointer)."""
    ptr = _load().miniocpp_alloc_aligned(size)
    if not ptr:
        raise RDMAError(f"miniocpp_alloc_aligned({size}) failed")
    return ptr


def free_aligned(ptr: int) -> None:
    _load().miniocpp_free_aligned(ctypes.c_void_p(ptr))


def _last_error(lib: ctypes.CDLL) -> str:
    msg = lib.miniocpp_last_error()
    return msg.decode(errors="replace") if msg else "unknown error"


def _raise(lib: ctypes.CDLL, rc: int, op: str) -> None:
    if rc == _ERR_RDMA_DECLINED:
        raise RDMADeclined(f"{op}: server declined RDMA: {_last_error(lib)}")
    if rc == _ERR_INVALID_ARG:
        raise RDMAError(f"{op}: invalid argument: {_last_error(lib)}")
    raise RDMAError(f"{op}: {_last_error(lib)}")


def _resolve(buf: BufferLike, length: Optional[int]) -> tuple[int, int, object]:
    """Return (pointer, length, keepalive) for a buffer or raw device pointer.

    The keepalive must outlive the call; for ``bytes`` we must copy into a
    mutable buffer, and losing the reference would free it mid-transfer.
    """
    if isinstance(buf, int):
        # Raw pointer (typically tensor.data_ptr()); the caller owns the memory
        # and must keep the owning tensor alive, so length is mandatory.
        if length is None or length <= 0:
            raise ValueError("length is required (and must be > 0) for a raw pointer")
        return buf, length, None

    if isinstance(buf, memoryview):
        if not buf.contiguous:
            raise ValueError("memoryview must be contiguous")
        owner = buf if not buf.readonly else bytearray(buf)
        arr = (ctypes.c_char * owner.nbytes).from_buffer(owner)
        return ctypes.addressof(arr), length or owner.nbytes, (owner, arr)

    if isinstance(buf, bytearray):
        arr = (ctypes.c_char * len(buf)).from_buffer(buf)
        return ctypes.addressof(arr), length or len(buf), (buf, arr)

    if isinstance(buf, bytes):
        arr = (ctypes.c_char * len(buf)).from_buffer_copy(buf)
        return ctypes.addressof(arr), length or len(buf), arr

    raise TypeError(
        f"unsupported buffer type {type(buf).__name__}; "
        "pass an int pointer, memoryview, bytearray, or bytes"
    )


def _check_size(size: int, op: str) -> None:
    """Refuse transfers at or above the cuObject registration limit.

    Passing >= 4 GiB does not error or fall back -- it segfaults inside
    libminiocpp. Turning that into an exception is the difference between a
    debuggable failure and a dead process mid-training.
    """
    if size >= MAX_REGISTRATION_BYTES:
        raise RDMAError(
            f"{op}: {size:,} bytes is at or above the cuObject registration limit "
            f"({MAX_REGISTRATION_BYTES:,}); libminiocpp segfaults rather than falling "
            f"back. Split the transfer into chunks of <= {SAFE_TRANSFER_BYTES:,} bytes."
        )


class RDMAClient:
    """A single ``miniocpp_client*``.

    Not thread-safe per instance; create one client per worker thread. Each
    client is cheap, but the underlying cuObjClient is process-wide.
    """

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        *,
        region: str = "",
        session_token: str = "",
        secure: bool = False,
    ) -> None:
        lib = _load()
        self._lib = lib
        self._handle = lib.miniocpp_client_new(
            endpoint.encode(),
            region.encode(),
            access_key.encode(),
            secret_key.encode(),
            session_token.encode(),
            1 if secure else 0,
        )
        if not self._handle:
            raise RDMAError(f"miniocpp_client_new({endpoint}): {_last_error(lib)}")

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._lib.miniocpp_client_free(ctypes.c_void_p(self._handle))
            self._handle = None

    def __enter__(self) -> "RDMAClient":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def __del__(self) -> None:
        # During interpreter shutdown the CUDA context and libminiocpp's own
        # globals may already be gone, and calling miniocpp_client_free() then
        # segfaults. Prefer explicit close(); if we get here at shutdown, leak the
        # handle rather than crash the process -- it is about to exit anyway.
        if sys.is_finalizing():
            return
        try:
            self.close()
        except Exception:
            pass

    def put(
        self, bucket: str, key: str, buf: BufferLike, length: Optional[int] = None
    ) -> tuple[int, str, str]:
        """Upload ``length`` bytes from ``buf``. Returns (bytes, etag, crc64nvme).

        ``buf`` may be a CUDA device pointer, in which case the payload is read
        straight out of VRAM by the NIC.
        """
        ptr, size, _keep = _resolve(buf, length)
        _check_size(size, "put")
        etag = (ctypes.c_char * 64)()
        checksum = (ctypes.c_char * 64)()
        rc = self._lib.miniocpp_put_object(
            ctypes.c_void_p(self._handle),
            bucket.encode(),
            key.encode(),
            ctypes.c_void_p(ptr),
            size,
            None,
            None,
            etag,
            checksum,
        )
        if rc < 0:
            _raise(self._lib, rc, f"put {bucket}/{key}")
        return int(rc), etag.value.decode(errors="replace"), checksum.value.decode(errors="replace")

    def get_range(self, bucket: str, key: str, buf: BufferLike, length: int,
                  offset: int) -> int:
        """Read ``length`` bytes starting at ``offset`` into ``buf``.

        Lets one registered device buffer be filled by several transfers, and lets
        an object larger than the 4 GiB registration limit be read in pieces --
        which is the only way to get a standard ~5 GB safetensors shard into VRAM
        over RDMA.
        """
        if not has_range():
            raise RDMAError(
                "this libminiocpp has no miniocpp_get_object_range; ranged RDMA GET "
                "requires minio-cpp PR #258. Without it, objects above "
                f"{MAX_REGISTRATION_BYTES:,} bytes cannot use RDMA."
            )
        ptr, size, _keep = _resolve(buf, length)
        _check_size(size, "get_range")
        rc = self._lib.miniocpp_get_object_range(
            ctypes.c_void_p(self._handle), bucket.encode(), key.encode(),
            ctypes.c_void_p(ptr), size, ctypes.c_uint64(offset),
        )
        if rc < 0:
            _raise(self._lib, rc, f"get_range {bucket}/{key}[{offset}:{offset + size}]")
        return int(rc)

    def get(self, bucket: str, key: str, buf: BufferLike, length: Optional[int] = None) -> int:
        """Download into ``buf``. Returns bytes transferred.

        ``buf`` may be a CUDA device pointer, in which case the NIC DMAs
        straight into VRAM with no host bounce.
        """
        ptr, size, _keep = _resolve(buf, length)
        _check_size(size, "get")
        rc = self._lib.miniocpp_get_object(
            ctypes.c_void_p(self._handle),
            bucket.encode(),
            key.encode(),
            ctypes.c_void_p(ptr),
            size,
            None,
            None,
        )
        if rc < 0:
            _raise(self._lib, rc, f"get {bucket}/{key}")
        return int(rc)
