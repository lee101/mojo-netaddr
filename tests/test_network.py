import itertools

import numpy as np
import pytest
import netaddr

import mojo_netaddr as mojo


@pytest.mark.parametrize(
    "text",
    [
        "0.0.0.0/0",
        "1.2.3.4/24",
        "10.0.0.0/31",
        "10.0.0.1/32",
        "::/0",
        "::1/64",
        "2001:db8::/32",
        "2001:db8::1/127",
        "ffff::/16",
    ],
)
def test_network_properties_parity(text):
    ours, theirs = mojo.IPNetwork(text), netaddr.IPNetwork(text)
    for name in ("value", "version", "prefixlen", "first", "last", "size"):
        assert getattr(ours, name) == getattr(theirs, name), name
    for name in ("ip", "network", "broadcast", "netmask", "hostmask", "cidr"):
        assert str(getattr(ours, name)) == str(getattr(theirs, name)), name
    assert ours.key() == theirs.key()
    assert ours.sort_key() == theirs.sort_key()
    assert str(ours) == str(theirs)
    assert repr(ours) == repr(theirs)


def test_network_netmask_hostmask_and_nohost():
    values = (
        "192.0.2.129/255.255.255.0",
        "192.0.2.129/0.0.0.255",
        "2001:db8::1/ffff:ffff:ffff:ffff::",
    )
    for value in values:
        assert str(mojo.IPNetwork(value)) == str(netaddr.IPNetwork(value))
    assert str(mojo.IPNetwork("192.0.2.129/24", flags=mojo.NOHOST)) == str(
        netaddr.IPNetwork("192.0.2.129/24", flags=netaddr.NOHOST)
    )


def test_network_index_iteration_and_hosts():
    ours, theirs = mojo.IPNetwork("192.0.2.0/29"), netaddr.IPNetwork("192.0.2.0/29")
    assert [str(ip) for ip in ours] == [str(ip) for ip in theirs]
    assert str(ours[-1]) == str(theirs[-1])
    assert [str(ip) for ip in ours[1:7:2]] == [str(ip) for ip in theirs[1:7:2]]
    assert [str(ip) for ip in ours.iter_hosts()] == [str(ip) for ip in theirs.iter_hosts()]


@pytest.mark.parametrize("text", ["192.0.2.0/31", "192.0.2.1/32", "2001:db8::/127"])
def test_small_network_hosts_parity(text):
    assert [str(ip) for ip in mojo.IPNetwork(text).iter_hosts()] == [
        str(ip) for ip in netaddr.IPNetwork(text).iter_hosts()
    ]


def test_subnet_supernet_and_neighbors_parity():
    ours, theirs = mojo.IPNetwork("192.0.2.64/26"), netaddr.IPNetwork("192.0.2.64/26")
    assert [str(net) for net in ours.subnet(29, count=5)] == [
        str(net) for net in theirs.subnet(29, count=5)
    ]
    assert [str(net) for net in ours.supernet(22)] == [
        str(net) for net in theirs.supernet(22)
    ]
    assert str(ours.next(2)) == str(theirs.next(2))
    assert str(ours.previous()) == str(theirs.previous())


def test_network_and_range_containment_parity():
    ours, theirs = mojo.IPNetwork("10.0.0.0/8"), netaddr.IPNetwork("10.0.0.0/8")
    candidates = (
        "10.2.3.4",
        "11.0.0.0",
        mojo.IPNetwork("10.1.0.0/16"),
        mojo.IPRange("10.1.2.3", "10.9.8.7"),
    )
    references = (
        "10.2.3.4",
        "11.0.0.0",
        netaddr.IPNetwork("10.1.0.0/16"),
        netaddr.IPRange("10.1.2.3", "10.9.8.7"),
    )
    assert [item in ours for item in candidates] == [item in theirs for item in references]


def test_compiled_bulk_containment_matches_scalar_and_upstream():
    values = [f"10.{a}.{b}.{c}" for a, b, c in itertools.product(range(8), repeat=3)]
    values += ["11.0.0.1", "192.168.1.1"]
    network = mojo.IPNetwork("10.0.0.0/8")
    result = network.contains_many(values)
    expected = np.array(
        [netaddr.IPAddress(value) in netaddr.IPNetwork("10.0.0.0/8") for value in values]
    )
    assert result.dtype == np.bool_
    assert np.array_equal(result, expected)


def test_bulk_containment_empty_input():
    result = mojo.IPNetwork("2001:db8::/32").contains_many([])
    assert result.dtype == np.bool_
    assert result.shape == (0,)


@pytest.mark.parametrize(
    "network,inside,outside",
    [
        ("10.0.0.0/9", 0x0A000001, 0x0A800001),
        (
            "2001:db8::/64",
            int(netaddr.IPAddress("2001:db8::1")),
            int(netaddr.IPAddress("2001:db8:0:1::1")),
        ),
    ],
)
def test_bulk_containment_simd_tail(network, inside, outside):
    ours = mojo.IPNetwork(network)
    values = [
        mojo.IPAddress(inside if index % 3 else outside, ours.version)
        for index in range(17)
    ]
    expected = np.array([index % 3 != 0 for index in range(17)])
    assert np.array_equal(ours.contains_many(values), expected)


@pytest.mark.parametrize("prefix", [0, 1, 8, 17, 31, 32])
def test_bulk_containment_ipv4_mask_boundaries_and_generator(prefix):
    network = mojo.IPNetwork((0xA5C37E19, prefix), 4).cidr
    values = [
        mojo.IPAddress((index * 2_654_435_761) & 0xFFFFFFFF, 4)
        for index in range(2 * 8 + 3)
    ]
    expected = np.array([value in network for value in values])
    assert np.array_equal(network.contains_many(value for value in values), expected)


def test_bulk_containment_large_chunk_boundaries():
    count = 524_291
    network = mojo.IPNetwork("10.0.0.0/9")
    values = [
        mojo.IPAddress(
            (0x0A000000 if index % 2 == 0 else 0x0B000000)
            + (index & 0xFFFFFF),
            4,
        )
        for index in range(count)
    ]
    expected = np.arange(count) % 2 == 0
    assert np.array_equal(network.contains_many(values), expected)


def test_range_properties_and_cidrs_parity():
    ours = mojo.IPRange("192.0.2.5", "192.0.3.17")
    theirs = netaddr.IPRange("192.0.2.5", "192.0.3.17")
    assert str(ours) == str(theirs)
    assert repr(ours) == repr(theirs)
    assert ours.first == theirs.first
    assert ours.last == theirs.last
    assert ours.size == theirs.size
    assert ours.key() == theirs.key()
    assert [str(net) for net in ours.cidrs()] == [str(net) for net in theirs.cidrs()]
