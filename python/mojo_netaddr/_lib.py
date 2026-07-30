from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src", "netaddr.mojo")
LIB = os.environ.get("MOJO_NETADDR_LIB") or os.path.join(
    ROOT, "dist", "libmojo-netaddr.so"
)

I = ctypes.c_int64
U = ctypes.c_uint64

_SIGNATURES = {
    "mna_range_to_cidrs": ([U, U, U, U, I, I, I, I, I], I),
    "mna_merge_ranges_v4": ([I, I, I, I, I, I], I),
    "mna_merge_ranges": ([I, I, I, I, I, I, I, I, I, I], I),
    "mna_contains_many_v4": ([U, I, I, I, I], I),
    "mna_contains_many": ([U, U, I, I, I, I, I, I], I),
}

_lib: ctypes.CDLL | None = None


def build() -> str:
    if not os.path.exists(LIB) or (
        os.path.exists(SRC) and os.path.getmtime(SRC) > os.path.getmtime(LIB)
    ):
        subprocess.run(
            ["bash", os.path.join(ROOT, "build", "build.sh")],
            check=True,
            cwd=ROOT,
        )
    return LIB


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_lib, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _lib


def addr(array: np.ndarray) -> int:
    if not isinstance(array, np.ndarray):
        raise TypeError("FFI buffers must be NumPy arrays")
    if array.dtype != np.uint64 and array.dtype != np.bool_:
        raise TypeError(f"unsupported FFI buffer dtype: {array.dtype}")
    if not array.flags.c_contiguous:
        raise ValueError("FFI buffers must be C-contiguous")
    if not array.flags.writeable:
        raise ValueError("FFI buffers must be writable")
    pointer = int(array.ctypes.data)
    if array.size and pointer == 0:
        raise RuntimeError("NumPy returned a null pointer for a non-empty buffer")
    return pointer


def halves(value: int) -> tuple[int, int]:
    return value >> 64, value & ((1 << 64) - 1)


def range_to_tuples(start: int, end: int, width: int) -> list[tuple[int, int]]:
    if width not in (32, 128):
        raise ValueError("address width must be 32 or 128")
    maximum = (1 << width) - 1
    if not 0 <= start <= end <= maximum:
        raise ValueError("range endpoints are outside the address width or reversed")
    sh, sl = halves(start)
    eh, el = halves(end)
    buffers = np.empty((3, 2 * width - 2), dtype=np.uint64)
    hi, lo, prefix = buffers
    fn = lib().mna_range_to_cidrs
    count = fn(
        sh,
        sl,
        eh,
        el,
        width,
        addr(hi),
        addr(lo),
        addr(prefix),
        buffers.shape[1],
    )
    if not 0 <= count <= buffers.shape[1]:
        raise RuntimeError(f"Mojo range kernel returned invalid count {count}")
    return [
        ((int(hi[i]) << 64) | int(lo[i]), int(prefix[i])) for i in range(count)
    ]


def merge_ranges(
    ranges: list[tuple[int, int]], width: int
) -> list[tuple[int, int]]:
    if width not in (32, 128):
        raise ValueError("address width must be 32 or 128")
    if not ranges:
        return []
    maximum = (1 << width) - 1
    if any(not 0 <= start <= end <= maximum for start, end in ranges):
        raise ValueError("merge range is outside the address width or reversed")
    if width <= 64:
        packed = np.asarray(ranges, dtype=np.uint64)
        scratch = np.empty(1, dtype=np.uint64)
        fn = lib().mna_merge_ranges_v4
        count = fn(
            addr(packed), len(ranges), addr(scratch), addr(scratch), 0, width
        )
        if count < 0:
            raise RuntimeError("Mojo IPv4 merge count pass failed")
        buffers = np.empty((2, count), dtype=np.uint64)
        lo, prefix = buffers
        written = fn(
            addr(packed), len(ranges), addr(lo), addr(prefix), count, width
        )
        if written != count:
            raise RuntimeError("Mojo CIDR merge returned an inconsistent result size")
        return [(int(lo[i]), int(prefix[i])) for i in range(count)]
    else:
        starts_hi = np.fromiter((a >> 64 for a, _ in ranges), dtype=np.uint64)
        starts_lo = np.fromiter(
            (a & ((1 << 64) - 1) for a, _ in ranges), dtype=np.uint64
        )
        ends_hi = np.fromiter((b >> 64 for _, b in ranges), dtype=np.uint64)
        ends_lo = np.fromiter(
            (b & ((1 << 64) - 1) for _, b in ranges), dtype=np.uint64
        )
    scratch = np.empty(1, dtype=np.uint64)
    fn = lib().mna_merge_ranges
    args = (
        addr(starts_hi),
        addr(starts_lo),
        addr(ends_hi),
        addr(ends_lo),
        len(ranges),
        width,
    )
    count = fn(*args, addr(scratch), addr(scratch), addr(scratch), 0)
    if count < 0:
        raise RuntimeError("Mojo IPv6 merge count pass failed")
    hi = np.empty(count, dtype=np.uint64)
    lo = np.empty(count, dtype=np.uint64)
    prefix = np.empty(count, dtype=np.uint64)
    written = fn(*args, addr(hi), addr(lo), addr(prefix), count)
    if written != count:
        raise RuntimeError("Mojo CIDR merge returned an inconsistent result size")
    return [
        ((int(hi[i]) << 64) | int(lo[i]), int(prefix[i])) for i in range(count)
    ]


def contains_many(
    first: int, prefix: int, width: int, values: list[int]
) -> np.ndarray:
    if width not in (32, 128):
        raise ValueError("address width must be 32 or 128")
    maximum = (1 << width) - 1
    if not 0 <= first <= maximum or not 0 <= prefix <= width:
        raise ValueError("invalid network passed to bulk containment")
    if any(not isinstance(value, int) or not 0 <= value <= maximum for value in values):
        raise ValueError("bulk containment value is outside the address width")
    if not values:
        return np.empty(0, dtype=np.bool_)
    if width == 32:
        values_lo = np.asarray(values, dtype=np.uint64)
        result = np.empty(len(values), dtype=np.bool_)
        status = lib().mna_contains_many_v4(
            first,
            prefix,
            addr(values_lo),
            len(values),
            addr(result),
        )
        if status != 0:
            raise RuntimeError(f"Mojo IPv4 containment kernel failed with status {status}")
        return result
    else:
        values_hi = np.fromiter((v >> 64 for v in values), dtype=np.uint64)
        values_lo = np.fromiter(
            (v & ((1 << 64) - 1) for v in values), dtype=np.uint64
        )
    result = np.empty(len(values), dtype=np.bool_)
    nh, nl = halves(first)
    status = lib().mna_contains_many(
        nh,
        nl,
        prefix,
        width,
        addr(values_hi),
        addr(values_lo),
        len(values),
        addr(result),
    )
    if status != 0:
        raise RuntimeError(f"Mojo IPv6 containment kernel failed with status {status}")
    return result
