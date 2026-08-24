"""128-bit IP interval kernels exposed through a small C ABI."""

from std.sys.info import simd_width_of as simdwidthof

comptime UPtr = UnsafePointer[UInt64, AnyOrigin[mut=True]]
comptime BPtr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime W = simdwidthof[DType.float64]()
comptime PARALLEL_CHUNK_SIZE = 65536


def less(a_hi: UInt64, a_lo: UInt64, b_hi: UInt64, b_lo: UInt64) -> Bool:
    return a_hi < b_hi or (a_hi == b_hi and a_lo < b_lo)


def less_equal(a_hi: UInt64, a_lo: UInt64, b_hi: UInt64, b_lo: UInt64) -> Bool:
    return a_hi < b_hi or (a_hi == b_hi and a_lo <= b_lo)


def trailing_zero_bits(hi: UInt64, lo: UInt64, width: Int) -> Int:
    if lo != 0:
        var x = lo
        var n = 0
        while (x & UInt64(1)) == 0:
            n += 1
            x >>= 1
        return min(n, width)
    if width <= 64:
        return width
    if hi != 0:
        var x = hi
        var n = 64
        while (x & UInt64(1)) == 0:
            n += 1
            x >>= 1
        return min(n, width)
    return width


def block_last(
    cur_hi: UInt64, cur_lo: UInt64, host_bits: Int
) -> Tuple[UInt64, UInt64]:
    if host_bits == 128:
        return UInt64.MAX, UInt64.MAX
    if host_bits > 64:
        var hi_mask = (UInt64(1) << UInt64(host_bits - 64)) - UInt64(1)
        return cur_hi | hi_mask, UInt64.MAX
    if host_bits == 64:
        return cur_hi, UInt64.MAX
    if host_bits == 0:
        return cur_hi, cur_lo
    var lo_mask = (UInt64(1) << UInt64(host_bits)) - UInt64(1)
    return cur_hi, cur_lo | lo_mask


def range_blocks(
    start_hi: UInt64,
    start_lo: UInt64,
    end_hi: UInt64,
    end_lo: UInt64,
    width: Int,
    dst_hi: UPtr,
    dst_lo: UPtr,
    dst_prefix: UPtr,
    capacity: Int,
    offset: Int = 0,
) -> Int:
    var cur_hi = start_hi
    var cur_lo = start_lo
    var count = 0
    while less_equal(cur_hi, cur_lo, end_hi, end_lo):
        var host_bits = trailing_zero_bits(cur_hi, cur_lo, width)
        var last_hi, last_lo = block_last(cur_hi, cur_lo, host_bits)
        while less(end_hi, end_lo, last_hi, last_lo):
            host_bits -= 1
            last_hi, last_lo = block_last(cur_hi, cur_lo, host_bits)

        if offset + count < capacity:
            dst_hi[offset + count] = cur_hi
            dst_lo[offset + count] = cur_lo
            dst_prefix[offset + count] = UInt64(width - host_bits)
        count += 1

        if last_hi == end_hi and last_lo == end_lo:
            break
        if host_bits == 64:
            cur_hi += UInt64(1)
        elif host_bits > 64:
            cur_hi += UInt64(1) << UInt64(host_bits - 64)
            cur_lo = UInt64(0)
        else:
            var old_lo = cur_lo
            cur_lo += UInt64(1) << UInt64(host_bits)
            if cur_lo < old_lo:
                cur_hi += UInt64(1)
    return count


@export("mna_range_to_cidrs")
def mna_range_to_cidrs(
    start_hi: UInt64,
    start_lo: UInt64,
    end_hi: UInt64,
    end_lo: UInt64,
    width: Int,
    dst_hi_addr: Int,
    dst_lo_addr: Int,
    dst_prefix_addr: Int,
    capacity: Int,
) abi("C") -> Int:
    if (width != 32 and width != 128) or capacity < 0:
        return -1
    if less(end_hi, end_lo, start_hi, start_lo):
        return -1
    if dst_hi_addr == 0 or dst_lo_addr == 0 or dst_prefix_addr == 0:
        return -1
    return range_blocks(
        start_hi,
        start_lo,
        end_hi,
        end_lo,
        width,
        UPtr(unsafe_from_address=dst_hi_addr),
        UPtr(unsafe_from_address=dst_lo_addr),
        UPtr(unsafe_from_address=dst_prefix_addr),
        capacity,
    )


@export("mna_merge_ranges")
def mna_merge_ranges(
    starts_hi_addr: Int,
    starts_lo_addr: Int,
    ends_hi_addr: Int,
    ends_lo_addr: Int,
    n: Int,
    width: Int,
    dst_hi_addr: Int,
    dst_lo_addr: Int,
    dst_prefix_addr: Int,
    capacity: Int,
) abi("C") -> Int:
    if (width != 32 and width != 128) or n <= 0 or capacity < 0:
        return -1
    if (
        starts_hi_addr == 0
        or starts_lo_addr == 0
        or ends_hi_addr == 0
        or ends_lo_addr == 0
        or dst_hi_addr == 0
        or dst_lo_addr == 0
        or dst_prefix_addr == 0
    ):
        return -1
    var starts_hi = UPtr(unsafe_from_address=starts_hi_addr)
    var starts_lo = UPtr(unsafe_from_address=starts_lo_addr)
    var ends_hi = UPtr(unsafe_from_address=ends_hi_addr)
    var ends_lo = UPtr(unsafe_from_address=ends_lo_addr)
    var dst_hi = UPtr(unsafe_from_address=dst_hi_addr)
    var dst_lo = UPtr(unsafe_from_address=dst_lo_addr)
    var dst_prefix = UPtr(unsafe_from_address=dst_prefix_addr)
    if n == 0:
        return 0

    var cur_start_hi = starts_hi[0]
    var cur_start_lo = starts_lo[0]
    var cur_end_hi = ends_hi[0]
    var cur_end_lo = ends_lo[0]
    var count = 0

    for i in range(1, n):
        var adjacent = False
        if width == 32:
            adjacent = cur_end_lo != UInt64(0xFFFFFFFF) and starts_hi[i] == 0 and (
                starts_lo[i] == cur_end_lo + UInt64(1)
            )
        elif cur_end_hi != UInt64.MAX or cur_end_lo != UInt64.MAX:
            var next_hi = cur_end_hi
            var next_lo = cur_end_lo + UInt64(1)
            if next_lo == 0:
                next_hi += UInt64(1)
            adjacent = starts_hi[i] == next_hi and starts_lo[i] == next_lo

        if less_equal(starts_hi[i], starts_lo[i], cur_end_hi, cur_end_lo) or adjacent:
            if less(cur_end_hi, cur_end_lo, ends_hi[i], ends_lo[i]):
                cur_end_hi = ends_hi[i]
                cur_end_lo = ends_lo[i]
        else:
            count += range_blocks(
                cur_start_hi,
                cur_start_lo,
                cur_end_hi,
                cur_end_lo,
                width,
                dst_hi,
                dst_lo,
                dst_prefix,
                capacity,
                count,
            )
            cur_start_hi = starts_hi[i]
            cur_start_lo = starts_lo[i]
            cur_end_hi = ends_hi[i]
            cur_end_lo = ends_lo[i]

    count += range_blocks(
        cur_start_hi,
        cur_start_lo,
        cur_end_hi,
        cur_end_lo,
        width,
        dst_hi,
        dst_lo,
        dst_prefix,
        capacity,
        count,
    )
    return count


def range_blocks_v4(
    start: UInt64,
    end: UInt64,
    dst_lo: UPtr,
    dst_prefix: UPtr,
    capacity: Int,
    offset: Int = 0,
) -> Int:
    var current = start
    var count = 0
    while current <= end:
        var host_bits = trailing_zero_bits(0, current, 32)
        var last_hi, last = block_last(0, current, host_bits)
        while end < last:
            host_bits -= 1
            last_hi, last = block_last(0, current, host_bits)
        if offset + count < capacity:
            dst_lo[offset + count] = current
            dst_prefix[offset + count] = UInt64(32 - host_bits)
        count += 1
        if last == end:
            break
        current = last + UInt64(1)
    return count


@export("mna_merge_ranges_v4")
def mna_merge_ranges_v4(
    ranges_addr: Int,
    n: Int,
    dst_lo_addr: Int,
    dst_prefix_addr: Int,
    capacity: Int,
    width: Int,
) abi("C") -> Int:
    if width != 32 or n <= 0 or capacity < 0:
        return -1
    if ranges_addr == 0 or dst_lo_addr == 0 or dst_prefix_addr == 0:
        return -1
    var ranges = UPtr(unsafe_from_address=ranges_addr)
    var dst_lo = UPtr(unsafe_from_address=dst_lo_addr)
    var dst_prefix = UPtr(unsafe_from_address=dst_prefix_addr)
    if n == 0:
        return 0

    var cur_start = ranges[0]
    var cur_end = ranges[1]
    var count = 0
    for i in range(1, n):
        var start = ranges[2 * i]
        var end = ranges[2 * i + 1]
        var adjacent = cur_end != UInt64(0xFFFFFFFF) and start == cur_end + UInt64(1)
        if start <= cur_end or adjacent:
            if cur_end < end:
                cur_end = end
        else:
            count += range_blocks_v4(
                cur_start, cur_end, dst_lo, dst_prefix, capacity, count
            )
            cur_start = start
            cur_end = end
    count += range_blocks_v4(
        cur_start, cur_end, dst_lo, dst_prefix, capacity, count
    )
    return count


@export("mna_contains_many")
def mna_contains_many(
    net_hi: UInt64,
    net_lo: UInt64,
    prefix: Int,
    width: Int,
    values_hi_addr: Int,
    values_lo_addr: Int,
    n: Int,
    dst_addr: Int,
) abi("C") -> Int:
    if (
        width != 128
        or prefix < 0
        or prefix > width
        or n <= 0
        or values_hi_addr == 0
        or values_lo_addr == 0
        or dst_addr == 0
    ):
        return -1
    var values_hi = UPtr(unsafe_from_address=values_hi_addr)
    var values_lo = UPtr(unsafe_from_address=values_lo_addr)
    var dst = BPtr(unsafe_from_address=dst_addr)
    var host_bits = width - prefix
    var last_hi, last_lo = block_last(net_hi, net_lo, host_bits)

    @parameter
    def process_chunk(chunk: Int):
        var chunk_start = chunk * PARALLEL_CHUNK_SIZE
        var chunk_end = min(chunk_start + PARALLEL_CHUNK_SIZE, n)
        var simd_end = chunk_end - ((chunk_end - chunk_start) % W)
        for i in range(chunk_start, simd_end, W):
            var value_hi = values_hi.load[width=W](i)
            var value_lo = values_lo.load[width=W](i)
            var above_start = value_hi.gt(net_hi) | (
                value_hi.eq(net_hi) & value_lo.ge(net_lo)
            )
            var below_end = value_hi.lt(last_hi) | (
                value_hi.eq(last_hi) & value_lo.le(last_lo)
            )
            dst.store(i, (above_start & below_end).cast[DType.uint8]())
        for i in range(simd_end, chunk_end):
            if less_equal(net_hi, net_lo, values_hi[i], values_lo[i]) and less_equal(
                values_hi[i], values_lo[i], last_hi, last_lo
            ):
                dst[i] = 1
            else:
                dst[i] = 0

    var chunks = (n + PARALLEL_CHUNK_SIZE - 1) // PARALLEL_CHUNK_SIZE
    for chunk in range(chunks):
        process_chunk(chunk)
    return 0


@export("mna_contains_many_v4")
def mna_contains_many_v4(
    net: UInt64,
    prefix: Int,
    values_addr: Int,
    n: Int,
    dst_addr: Int,
) abi("C") -> Int:
    if (
        prefix < 0
        or prefix > 32
        or n <= 0
        or values_addr == 0
        or dst_addr == 0
    ):
        return -1
    var values = UPtr(unsafe_from_address=values_addr)
    var dst = BPtr(unsafe_from_address=dst_addr)
    var host_bits = 32 - prefix
    var last_hi, last = block_last(0, net, host_bits)
    var mask = UInt64(0xFFFFFFFF) ^ (last - net)

    @parameter
    def process_chunk(chunk: Int):
        var chunk_start = chunk * PARALLEL_CHUNK_SIZE
        var chunk_end = min(chunk_start + PARALLEL_CHUNK_SIZE, n)
        var simd_end = chunk_end - ((chunk_end - chunk_start) % W)
        for i in range(chunk_start, simd_end, W):
            var value = values.load[width=W](i)
            dst.store(
                i,
                ((value & mask).eq(net)).cast[DType.uint8](),
            )
        for i in range(simd_end, chunk_end):
            dst[i] = UInt8((values[i] & mask) == net)

    var chunks = (n + PARALLEL_CHUNK_SIZE - 1) // PARALLEL_CHUNK_SIZE
    for chunk in range(chunks):
        process_chunk(chunk)
    return 0
