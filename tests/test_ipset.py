import netaddr

import mojo_netaddr as mojo


def strings(values):
    return [str(value) for value in values]


def test_ipset_construction_membership_and_size_parity():
    values = [
        "10.0.0.0/9",
        "10.128.0.0/9",
        "192.0.2.1",
        "2001:db8::/126",
    ]
    ours, theirs = mojo.IPSet(values), netaddr.IPSet(values)
    assert strings(ours.iter_cidrs()) == strings(theirs.iter_cidrs())
    assert ours.size == theirs.size
    for value in ("10.1.2.3", "11.0.0.0", "192.0.2.1", "2001:db8::2"):
        assert (value in ours) == (value in theirs)


def test_ipset_add_remove_update_parity():
    ours, theirs = mojo.IPSet(["10.0.0.0/24"]), netaddr.IPSet(["10.0.0.0/24"])
    for value in ("10.0.1.0/24", "192.0.2.0/28"):
        ours.add(value)
        theirs.add(value)
    ours.remove("10.0.0.64/26")
    theirs.remove("10.0.0.64/26")
    ours.update(["198.51.100.0/30", "198.51.100.4/30"])
    theirs.update(["198.51.100.0/30", "198.51.100.4/30"])
    assert strings(ours.iter_cidrs()) == strings(theirs.iter_cidrs())
    assert ours.size == theirs.size


def test_ipset_algebra_parity():
    left_values = ["10.0.0.0/8", "192.0.2.0/24"]
    right_values = ["10.0.0.0/9", "198.51.100.0/24"]
    ours_left, ours_right = mojo.IPSet(left_values), mojo.IPSet(right_values)
    their_left, their_right = netaddr.IPSet(left_values), netaddr.IPSet(right_values)
    operations = (
        ("union", ours_left.union(ours_right), their_left.union(their_right)),
        (
            "intersection",
            ours_left.intersection(ours_right),
            their_left.intersection(their_right),
        ),
        ("difference", ours_left.difference(ours_right), their_left.difference(their_right)),
        (
            "symmetric_difference",
            ours_left.symmetric_difference(ours_right),
            their_left.symmetric_difference(their_right),
        ),
    )
    for name, ours, theirs in operations:
        assert strings(ours.iter_cidrs()) == strings(theirs.iter_cidrs()), name
    assert ours_right.issubset(ours_left) == their_right.issubset(their_left)
    assert ours_left.isdisjoint(ours_right) == their_left.isdisjoint(their_right)


def test_ipset_ranges_and_contiguity_parity():
    values = ["192.0.2.0/25", "192.0.2.128/26", "192.0.2.192/26"]
    ours, theirs = mojo.IPSet(values), netaddr.IPSet(values)
    assert ours.iscontiguous() == theirs.iscontiguous()
    assert strings(ours.iter_ipranges()) == strings(theirs.iter_ipranges())
    assert str(ours.iprange()) == str(theirs.iprange())
