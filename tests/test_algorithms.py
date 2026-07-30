import random

import pytest
import netaddr

import mojo_netaddr as mojo


@pytest.mark.parametrize(
    "version,start,end",
    [
        (4, 0, (1 << 32) - 1),
        (4, 1, (1 << 32) - 2),
        (4, 0xC0000205, 0xC0000311),
        (6, 0, (1 << 128) - 1),
        (6, 1, (1 << 128) - 2),
        (6, int(netaddr.IPAddress("2001:db8::1")), int(netaddr.IPAddress("2001:db8::ffff"))),
    ],
)
def test_compiled_range_to_cidrs_parity(version, start, end):
    ours = mojo.iprange_to_cidrs(
        mojo.IPAddress(start, version), mojo.IPAddress(end, version)
    )
    theirs = netaddr.iprange_to_cidrs(
        netaddr.IPAddress(start, version), netaddr.IPAddress(end, version)
    )
    assert [str(net) for net in ours] == [str(net) for net in theirs]


def test_random_range_decomposition_parity():
    rng = random.Random(7)
    for version, width in ((4, 32), (6, 128)):
        for _ in range(100):
            start = rng.getrandbits(width)
            end = rng.randrange(start, 1 << width)
            ours = mojo.iprange_to_cidrs(
                mojo.IPAddress(start, version), mojo.IPAddress(end, version)
            )
            theirs = netaddr.iprange_to_cidrs(
                netaddr.IPAddress(start, version), netaddr.IPAddress(end, version)
            )
            assert [str(net) for net in ours] == [str(net) for net in theirs]


def test_reversed_range_is_rejected_before_ffi():
    with pytest.raises(mojo.AddrFormatError):
        mojo.iprange_to_cidrs("192.0.2.2", "192.0.2.1")


def test_compiled_cidr_merge_parity():
    inputs = [
        "10.0.0.0/25",
        "10.0.0.128/25",
        "10.0.1.0/24",
        "10.0.0.64/26",
        "192.0.2.1/32",
        "192.0.2.2/31",
        "2001:db8::/126",
        "2001:db8::4/126",
    ]
    assert [str(net) for net in mojo.cidr_merge(inputs)] == [
        str(net) for net in netaddr.cidr_merge(inputs)
    ]


def test_random_cidr_merge_parity():
    rng = random.Random(11)
    for version, width in ((4, 32), (6, 128)):
        tuples = []
        for _ in range(500):
            prefix = rng.randrange(width + 1)
            value = rng.getrandbits(width)
            value = value >> (width - prefix) << (width - prefix) if prefix else 0
            tuples.append((value, prefix))
        ours = mojo.cidr_merge(mojo.IPNetwork(item, version) for item in tuples)
        theirs = netaddr.cidr_merge(netaddr.IPNetwork(item, version) for item in tuples)
        assert [str(net) for net in ours] == [str(net) for net in theirs]


@pytest.mark.parametrize(
    "target,exclude",
    [
        ("192.0.2.0/24", "192.0.2.1"),
        ("10.0.0.0/8", "10.64.0.0/10"),
        ("2001:db8::/120", "2001:db8::40/124"),
        ("192.0.2.0/24", "198.51.100.0/24"),
        ("192.0.2.0/24", "0.0.0.0/0"),
    ],
)
def test_cidr_exclude_parity(target, exclude):
    assert [str(net) for net in mojo.cidr_exclude(target, exclude)] == [
        str(net) for net in netaddr.cidr_exclude(target, exclude)
    ]


def test_spanning_and_matching_parity():
    values = ["192.0.2.1", "192.0.3.0/25", "192.0.7.255"]
    assert str(mojo.spanning_cidr(values)) == str(netaddr.spanning_cidr(values))
    cidrs = ["0.0.0.0/0", "10.0.0.0/8", "10.2.0.0/16", "10.2.3.0/24"]
    for function in (
        "all_matching_cidrs",
        "smallest_matching_cidr",
        "largest_matching_cidr",
    ):
        ours = getattr(mojo, function)("10.2.3.4", cidrs)
        theirs = getattr(netaddr, function)("10.2.3.4", cidrs)
        if isinstance(ours, list):
            assert [str(net) for net in ours] == [str(net) for net in theirs]
        else:
            assert str(ours) == str(theirs)


@pytest.mark.parametrize("value", ["10", "10/16", "128", "192.168", "224.1/24", "::1"])
def test_cidr_abbreviation_parity(value):
    assert mojo.cidr_abbrev_to_verbose(value) == netaddr.cidr_abbrev_to_verbose(value)


def test_iter_iprange_parity():
    for step in (1, 7, -3):
        start, end = ("192.0.2.1", "192.0.2.20") if step > 0 else (
            "192.0.2.20",
            "192.0.2.1",
        )
        assert [str(ip) for ip in mojo.iter_iprange(start, end, step)] == [
            str(ip) for ip in netaddr.iter_iprange(start, end, step)
        ]
