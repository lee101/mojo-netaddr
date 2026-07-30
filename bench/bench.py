from __future__ import annotations

import gc
import math
import os
import platform
import sys
import time

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python")
)

import netaddr

import mojo_netaddr as mojo


def best_time(fn, repeat=3):
    best = math.inf
    for _ in range(repeat):
        gc.collect()
        gc.disable()
        start = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - start
        gc.enable()
        best = min(best, elapsed)
    return best


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as source:
            for line in source:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main():
    cases = []

    start4_m, end4_m = mojo.IPAddress(1, 4), mojo.IPAddress((1 << 32) - 2, 4)
    start4_n, end4_n = netaddr.IPAddress(1, 4), netaddr.IPAddress((1 << 32) - 2, 4)

    def mojo_range4():
        for _ in range(10_000):
            mojo.iprange_to_cidrs(start4_m, end4_m)

    def netaddr_range4():
        for _ in range(10_000):
            netaddr.iprange_to_cidrs(start4_n, end4_n)

    cases.append(("IPv4 range to CIDRs (10k calls)", mojo_range4, netaddr_range4))

    start6_m, end6_m = mojo.IPAddress(1, 6), mojo.IPAddress((1 << 128) - 2, 6)
    start6_n, end6_n = netaddr.IPAddress(1, 6), netaddr.IPAddress((1 << 128) - 2, 6)

    def mojo_range6():
        for _ in range(2_000):
            mojo.iprange_to_cidrs(start6_m, end6_m)

    def netaddr_range6():
        for _ in range(2_000):
            netaddr.iprange_to_cidrs(start6_n, end6_n)

    cases.append(("IPv6 range to CIDRs (2k calls)", mojo_range6, netaddr_range6))

    merge_count = 300_000
    mojo_nets = [
        mojo.IPNetwork((0x0A000000 + index, 32), 4) for index in range(merge_count)
    ]
    netaddr_nets = [
        netaddr.IPNetwork((0x0A000000 + index, 32), 4)
        for index in range(merge_count)
    ]
    cases.append(
        (
            "Merge 300k adjacent IPv4 /32s",
            lambda: mojo.cidr_merge(mojo_nets),
            lambda: netaddr.cidr_merge(netaddr_nets),
        )
    )

    membership_count = 500_000
    values = [
        0x0A000000 + (index * 2_654_435_761 & 0xFFFFFF)
        for index in range(membership_count)
    ]
    mojo_ips = [mojo.IPAddress(value, 4) for value in values]
    netaddr_ips = [netaddr.IPAddress(value, 4) for value in values]
    mojo_network = mojo.IPNetwork("10.0.0.0/9")
    netaddr_network = netaddr.IPNetwork("10.0.0.0/9")
    cases.append(
        (
            "Membership of 500k IPv4 addresses",
            lambda: mojo_network.contains_many(mojo_ips),
            lambda: [ip in netaddr_network for ip in netaddr_ips],
        )
    )

    print(f"Machine: {cpu_name()} ({platform.machine()}, {platform.system()})")
    print(f"Python {platform.python_version()}, netaddr {netaddr.__version__}")
    print()
    print("| case | mojo-netaddr | netaddr | result |")
    print("| --- | ---: | ---: | ---: |")
    for name, ours, theirs in cases:
        ours()
        theirs()
        mojo_seconds = best_time(ours)
        netaddr_seconds = best_time(theirs)
        ratio = netaddr_seconds / mojo_seconds
        result = f"{ratio:.2f}x faster" if ratio >= 1 else f"{1 / ratio:.2f}x slower"
        print(
            f"| {name} | {mojo_seconds * 1e3:.1f} ms | "
            f"{netaddr_seconds * 1e3:.1f} ms | {result} |"
        )


if __name__ == "__main__":
    main()
