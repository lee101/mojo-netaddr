import pickle

import pytest
import netaddr

import mojo_netaddr as mojo


@pytest.mark.parametrize(
    "text",
    [
        "0.0.0.0",
        "1.2.3.4",
        "127.0.0.1",
        "192.0.2.255",
        "255.255.255.255",
        "::",
        "::1",
        "::ffff:192.0.2.1",
        "2001:db8::dead:beef",
        "fe80::1",
        "ff02::1",
        "ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff",
    ],
)
def test_address_representation_parity(text):
    ours, theirs = mojo.IPAddress(text), netaddr.IPAddress(text)
    assert str(ours) == str(theirs)
    assert repr(ours) == repr(theirs)
    assert int(ours) == int(theirs)
    assert ours.value == theirs.value
    assert ours.version == theirs.version
    assert ours.key() == theirs.key()
    assert ours.sort_key() == theirs.sort_key()
    assert ours.packed == theirs.packed
    assert ours.words == theirs.words
    assert ours.bits() == theirs.bits()
    assert ours.bits("-") == theirs.bits("-")
    assert ours.bin == theirs.bin
    assert ours.reverse_dns == theirs.reverse_dns


@pytest.mark.parametrize(
    "text",
    [
        "0.0.0.0",
        "255.255.255.0",
        "0.0.0.255",
        "127.0.0.1",
        "169.254.10.2",
        "192.168.2.1",
        "192.0.2.1",
        "224.1.2.3",
        "::",
        "::1",
        "::ffff:10.0.0.1",
        "2001:db8::1",
        "fc00::1",
        "fe80::1",
        "ff00::1",
    ],
)
def test_address_predicate_parity(text):
    ours, theirs = mojo.IPAddress(text), netaddr.IPAddress(text)
    methods = (
        "is_netmask",
        "is_hostmask",
        "netmask_bits",
        "is_multicast",
        "is_unicast",
        "is_loopback",
        "is_link_local",
        "is_reserved",
        "is_ipv4_mapped",
        "is_ipv4_compat",
        "is_global",
    )
    for name in methods:
        assert getattr(ours, name)() == getattr(theirs, name)(), name


def test_address_arithmetic_and_bitwise_parity():
    ours, theirs = mojo.IPAddress("192.0.2.17"), netaddr.IPAddress("192.0.2.17")
    for operation in (
        lambda value: value + 13,
        lambda value: value - 13,
        lambda value: value | 0xF0,
        lambda value: value & 0xFFFFFF00,
        lambda value: value ^ 0x1234,
        lambda value: value >> 3,
    ):
        assert str(operation(ours)) == str(operation(theirs))
    with pytest.raises(IndexError):
        mojo.IPAddress("255.255.255.255") + 1
    with pytest.raises(mojo.AddrFormatError):
        ours << 1
    with pytest.raises(netaddr.AddrFormatError):
        theirs << 1


def test_address_conversion_parity():
    for text in ("0.0.0.0", "192.0.2.1", "255.255.255.255"):
        ours, theirs = mojo.IPAddress(text), netaddr.IPAddress(text)
        assert str(ours.ipv6()) == str(theirs.ipv6())
        assert str(ours.ipv6(True)) == str(theirs.ipv6(True))
        assert str(ours.ipv6().ipv4()) == str(theirs.ipv6().ipv4())
    assert str(mojo.IPAddress("::ffff:192.0.2.1").to_canonical()) == "192.0.2.1"


def test_zerofill_and_validation_parity():
    text = "010.020.030.040"
    assert str(mojo.IPAddress(text, flags=mojo.ZEROFILL)) == str(
        netaddr.IPAddress(text, flags=netaddr.ZEROFILL)
    )
    for value in ("192.0.2.1", "999.1.1.1", "::1", "bad"):
        assert mojo.valid_ipv4(value) == netaddr.valid_ipv4(value)
        assert mojo.valid_ipv6(value) == netaddr.valid_ipv6(value)


def test_address_pickle_roundtrip():
    value = mojo.IPAddress("2001:db8::1234")
    assert pickle.loads(pickle.dumps(value)) == value
