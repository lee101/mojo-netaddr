# mojo-netaddr

`mojo-netaddr` is a standalone port of the compute-heavy IP address and
network-math subset of [netaddr](https://netaddr.readthedocs.io/). It keeps the
covered class names, function names, signatures, object formatting, and
IPv4/IPv6 behavior, with 128-bit interval operations implemented in Mojo.

Use `import mojo_netaddr as netaddr` when migrating covered code. The separate
module name lets the real Python package remain installed for parity tests.

```python
import mojo_netaddr as netaddr

network = netaddr.IPNetwork("10.0.0.0/8")
print(network.contains_many(["10.1.2.3", "192.0.2.1"]).tolist())

blocks = netaddr.cidr_merge(
    ["192.0.2.0/25", "192.0.2.128/25", "198.51.100.0/24"]
)
print([str(block) for block in blocks])
```

Output:

```text
[True, False]
['192.0.2.0/24', '198.51.100.0/24']
```

`contains_many()` is the one deliberate extension to the upstream API. It
amortizes one FFI call over a large address sequence.

## Tested compatibility subset

The test suite compares the following behavior with upstream netaddr:

| area | tested behavior |
| --- | --- |
| addresses | `IPAddress`; IPv4 and IPv6 parsing and formatting, integer/packed/word/binary forms, arithmetic and bitwise operations, masks, reverse DNS names, mapped/compatible conversion, validation, and the classification methods exercised in `tests/test_address.py` |
| networks | `IPNetwork`; CIDR, netmask and hostmask input, preserved host bits, properties and formatting, indexing, iteration, containment, hosts, subnet/supernet traversal, and adjacent networks |
| ranges | `IPRange`, `iter_iprange`, and `iprange_to_cidrs` |
| CIDR math | `cidr_merge`, `cidr_exclude`, `spanning_cidr`, `cidr_abbrev_to_verbose`, matching-CIDR helpers |
| sets | `IPSet`; construction, compaction through mutation, membership, ranges, union, intersection, difference, symmetric difference, subset checks, disjointness, and contiguity |
| flags | `ZEROFILL` on `IPAddress` and `NOHOST` on `IPNetwork` |

The compiled kernels cover exact IPv4 and IPv6 range decomposition, sorted
interval merging and bulk network membership. IPv6 values are never narrowed:
Mojo represents each as a `(high UInt64, low UInt64)` pair.

The package also exports `iter_unique_ips`, `expand_partial_ipv4_address`,
`INET_PTON`, and `INET_ATON`, but they are not part of the tested compatibility
contract yet. Not covered are the rest of upstream netaddr, including
hardware-address and registry features (`EUI`, `OUI`, `IAB`), IANA metadata
through `.info`, IPv4 glob and Nmap-range helpers, RFC base85 conversion,
custom IPv6 formatting dialects, and `SubnetSplitter`.

## Install and run

The Pixi environment pins the tested Mojo nightly and installs netaddr 1.3.0
for behavioral comparison.

```bash
pixi install
pixi run build
pixi run python - <<'PY'
import mojo_netaddr as netaddr

network = netaddr.IPNetwork("10.0.0.0/8")
print(network.contains_many(["10.1.2.3", "192.0.2.1"]).tolist())
PY
```

`pixi run build` creates `dist/libmojo-netaddr.so`. Imports rebuild it if the
Mojo source is newer, and `MOJO_NETADDR_LIB=/path/to/library.so` can point the
wrapper at a prebuilt library.

For development, run `pixi run test` and `pixi run bench`.

## Performance

Measured through the flocked `pixi run bench` task on an Intel Xeon E5-2697 v4
at 2.30 GHz, x86-64 Linux, with Python 3.13.14 and netaddr 1.3.0. These are the
best of three warm runs.

| case | mojo-netaddr | netaddr | result |
| --- | ---: | ---: | ---: |
| IPv4 range to CIDRs (10k calls) | 738.0 ms | 1290.9 ms | 1.75x faster |
| IPv6 range to CIDRs (2k calls) | 600.0 ms | 897.0 ms | 1.49x faster |
| Merge 300k adjacent IPv4 /32s | 318.1 ms | 440.9 ms | 1.39x faster |
| Membership of 500k IPv4 addresses | 164.7 ms | 166.5 ms | 1.01x faster |

Range decomposition uses one FFI crossing and one caller-owned NumPy allocation
per call. Internal result construction skips redundant validation of values
already produced by the compiled kernel. The IPv4 merge path consumes packed
interval pairs directly, avoiding high-word arrays and column copies. Bulk
membership uses native-width SIMD with a scalar remainder, writes directly
into its boolean NumPy result, and switches to thresholded CPU parallelism for
large inputs with a serial fallback.

No GPU path is provided.

## How it works

`src/netaddr.mojo` is one compilation unit exposing C ABI entry points. Python
owns every contiguous, native-dtype NumPy allocation and keeps it alive for
the synchronous ctypes call. The wrapper validates widths, bounds, capacity,
dtype, contiguity, writability, and non-null addresses before crossing the
boundary. Mojo reconstructs pointers only after validating exported arguments;
it does not allocate or retain Python memory.

IPv6 uses a structure-of-arrays layout with separate contiguous high-word and
low-word buffers. IPv4 merge input is a packed array of `(start, end)` pairs.
Merge kernels run once to count the exact output size and again to fill
caller-owned storage. Range decomposition chooses the largest aligned
power-of-two block that stays inside each interval. CIDR merge first coalesces
sorted overlapping or adjacent intervals, then applies the same exact
decomposition.

The Python layer handles rich objects, strings, exceptions and iteration.
Tests compare those observable behaviors directly with the real netaddr
package, including randomized IPv4 and IPv6 vectors.

## License

MIT
