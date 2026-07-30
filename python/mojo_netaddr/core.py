from __future__ import annotations

import ipaddress
import socket
import sys
from itertools import chain
from typing import Iterable, Iterator

import numpy as np

from . import _lib

INET_PTON = 1
ZEROFILL = 2
NOHOST = 4
INET_ATON = 8


class AddrFormatError(Exception):
    pass


class AddrConversionError(Exception):
    pass


class NotRegisteredError(Exception):
    pass


def _width(version: int) -> int:
    return 32 if version == 4 else 128


def _maximum(version: int) -> int:
    return (1 << _width(version)) - 1


def _parse_text(text: str, version: int | None, flags: int) -> tuple[int, int]:
    candidate = text
    if flags & ZEROFILL and "." in candidate and ":" not in candidate:
        pieces = candidate.split(".")
        if len(pieces) == 4:
            try:
                candidate = ".".join(str(int(piece, 10)) for piece in pieces)
            except ValueError:
                pass
    if flags & INET_ATON and ":" not in candidate:
        try:
            packed = socket.inet_aton(candidate)
            value = int.from_bytes(packed, "big")
            if version not in (None, 4):
                raise AddrFormatError(f"base address {text!r} is not IPv{version}")
            return value, 4
        except OSError as exc:
            raise AddrFormatError(f"{text!r} is not a valid IPv4 address string!") from exc
    try:
        parsed = ipaddress.ip_address(candidate)
    except ValueError as exc:
        raise AddrFormatError(f"failed to detect a valid IP address from {text!r}") from exc
    if version is not None and parsed.version != version:
        raise AddrFormatError(f"base address {text!r} is not IPv{version}")
    return int(parsed), parsed.version


class BaseIP:
    __slots__ = ("_value", "_version")

    @property
    def value(self) -> int:
        return self._value

    @value.setter
    def value(self, value: int) -> None:
        if not isinstance(value, int):
            raise TypeError(f"int argument expected, not {type(value)}")
        if not 0 <= value <= _maximum(self.version):
            raise AddrFormatError(f"value out of bounds for an IPv{self.version} address!")
        self._value = value

    @property
    def version(self) -> int:
        return self._version

    def key(self):
        raise NotImplementedError

    def sort_key(self):
        raise NotImplementedError

    def __hash__(self):
        return hash(self.key())

    def __eq__(self, other):
        try:
            return self.key() == other.key()
        except (AttributeError, TypeError):
            return NotImplemented

    def __lt__(self, other):
        try:
            return self.sort_key() < other.sort_key()
        except (AttributeError, TypeError):
            return NotImplemented

    def __le__(self, other):
        try:
            return self.sort_key() <= other.sort_key()
        except (AttributeError, TypeError):
            return NotImplemented

    def __gt__(self, other):
        try:
            return self.sort_key() > other.sort_key()
        except (AttributeError, TypeError):
            return NotImplemented

    def __ge__(self, other):
        try:
            return self.sort_key() >= other.sort_key()
        except (AttributeError, TypeError):
            return NotImplemented

    def _bounds(self) -> tuple[int, int]:
        if isinstance(self, IPAddress):
            return self.value, self.value
        return self.first, self.last

    def _within(self, cidr: str) -> bool:
        network = ipaddress.ip_network(cidr)
        if self.version != network.version:
            return False
        first, last = self._bounds()
        return int(network.network_address) <= first and last <= int(network.broadcast_address)

    def is_unicast(self) -> bool:
        return not self.is_multicast()

    def is_multicast(self) -> bool:
        return self._within("224.0.0.0/4" if self.version == 4 else "ff00::/8")

    def is_loopback(self) -> bool:
        return self._within("127.0.0.0/8" if self.version == 4 else "::1/128")

    def is_link_local(self) -> bool:
        return self._within("169.254.0.0/16" if self.version == 4 else "fe80::/10")

    def is_ipv4_mapped(self) -> bool:
        first, last = self._bounds()
        return self.version == 6 and first >> 32 == 0xFFFF and last >> 32 == 0xFFFF

    def is_ipv4_compat(self) -> bool:
        first, last = self._bounds()
        return self.version == 6 and first >> 32 == 0 and last >> 32 == 0

    def is_reserved(self) -> bool:
        if self.version == 4:
            ranges = (
                "0.0.0.0/8",
                "127.0.0.0/8",
                "192.0.2.0/24",
                "192.88.99.0/24",
                "198.51.100.0/24",
                "203.0.113.0/24",
                "225.0.0.0/8",
                "226.0.0.0/7",
                "228.0.0.0/6",
                "233.252.0.0/24",
                "234.0.0.0/7",
                "236.0.0.0/7",
                "238.0.0.0/8",
                "240.0.0.0/4",
            )
        else:
            ranges = (
                "ff00::/12",
                "::/8",
                "100::/8",
                "200::/7",
                "400::/6",
                "800::/5",
                "1000::/4",
                "4000::/2",
                "8000::/2",
                "c000::/3",
                "e000::/4",
                "f000::/5",
                "f800::/6",
                "fe00::/9",
            )
        return any(self._within(cidr) for cidr in ranges)


class IPAddress(BaseIP):
    __slots__ = ()

    def __init__(self, addr, version=None, flags=0):
        if flags & ~(INET_PTON | ZEROFILL | INET_ATON):
            raise ValueError(f"Unrecognized IPAddress flags value: {flags}")
        if flags & INET_ATON and flags & INET_PTON:
            raise ValueError("INET_ATON and INET_PTON are mutually exclusive")
        if version not in (None, 4, 6):
            raise ValueError(f"{version!r} is an invalid IP version!")
        if isinstance(addr, BaseIP):
            if version is not None and version != addr.version:
                raise ValueError("cannot switch IP versions using copy constructor!")
            self._value, self._version = addr.value, addr.version
        elif isinstance(addr, str):
            if "/" in addr:
                raise ValueError(
                    "IPAddress() does not support netmasks or subnet prefixes!"
                )
            self._value, self._version = _parse_text(addr, version, flags)
        elif isinstance(addr, (bytes, bytearray)):
            if len(addr) not in (4, 16):
                raise AddrFormatError(f"failed to detect a valid IP address from {addr!r}")
            detected = 4 if len(addr) == 4 else 6
            if version is not None and version != detected:
                raise AddrFormatError(f"bad address format: {addr!r}")
            self._value, self._version = int.from_bytes(addr, "big"), detected
        else:
            try:
                value = int(addr)
            except (TypeError, ValueError) as exc:
                raise AddrFormatError(
                    f"failed to detect a valid IP address from {addr!r}"
                ) from exc
            detected = version or (4 if 0 <= value <= (1 << 32) - 1 else 6)
            if not 0 <= value <= _maximum(detected):
                raise AddrFormatError(f"bad address format: {addr!r}")
            self._value, self._version = value, detected

    def key(self):
        return self.version, self.value

    def sort_key(self):
        return self.version, self.value, _width(self.version)

    def __int__(self):
        return self.value

    def __index__(self):
        return self.value

    def __bytes__(self):
        return self.packed

    def __bool__(self):
        return bool(self.value)

    def __str__(self):
        if self.version == 4:
            return str(ipaddress.IPv4Address(self.value))
        if 0x10000 <= self.value <= 0xFFFFFFFF:
            return f"::{ipaddress.IPv4Address(self.value)}"
        if 0xFFFF00000000 <= self.value <= 0xFFFFFFFFFFFF:
            return f"::ffff:{ipaddress.IPv4Address(self.value - 0xFFFF00000000)}"
        return str(ipaddress.IPv6Address(self.value))

    def __repr__(self):
        return f"IPAddress('{self}')"

    def _new_arithmetic(self, value: int):
        if not 0 <= value <= _maximum(self.version):
            raise IndexError("result outside valid IP address boundary!")
        return IPAddress(value, self.version)

    def __add__(self, num):
        return self._new_arithmetic(self.value + num)

    __radd__ = __add__

    def __sub__(self, num):
        return self._new_arithmetic(self.value - num)

    def __rsub__(self, num):
        return self._new_arithmetic(num - self.value)

    def __iadd__(self, num):
        self.value = self._new_arithmetic(self.value + num).value
        return self

    def __isub__(self, num):
        self.value = self._new_arithmetic(self.value - num).value
        return self

    def __or__(self, other):
        return IPAddress(self.value | int(other), self.version)

    def __and__(self, other):
        return IPAddress(self.value & int(other), self.version)

    def __xor__(self, other):
        return IPAddress(self.value ^ int(other), self.version)

    def __lshift__(self, bits):
        return IPAddress(self.value << bits, self.version)

    def __rshift__(self, bits):
        return IPAddress(self.value >> bits, self.version)

    @property
    def packed(self):
        return self.value.to_bytes(_width(self.version) // 8, "big")

    @property
    def words(self):
        word_bits = 8 if self.version == 4 else 16
        count = _width(self.version) // word_bits
        mask = (1 << word_bits) - 1
        return tuple(
            (self.value >> (word_bits * (count - i - 1))) & mask for i in range(count)
        )

    def bits(self, word_sep=None):
        word_bits = 8 if self.version == 4 else 16
        separator = ("." if self.version == 4 else ":") if word_sep is None else word_sep
        return separator.join(f"{word:0{word_bits}b}" for word in self.words)

    @property
    def bin(self):
        return bin(self.value)

    @property
    def reverse_dns(self):
        cls = ipaddress.IPv4Address if self.version == 4 else ipaddress.IPv6Address
        return cls(self.value).reverse_pointer + "."

    def format(self, dialect=None):
        if dialect is not None and not hasattr(dialect, "word_fmt"):
            raise TypeError("custom dialects should subclass ipv6_verbose!")
        return str(self)

    def is_netmask(self):
        inverted = (self.value ^ _maximum(self.version)) + 1
        return inverted & (inverted - 1) == 0

    def is_hostmask(self):
        incremented = self.value + 1
        return incremented & (incremented - 1) == 0

    def netmask_bits(self):
        if not self.is_netmask():
            return _width(self.version)
        return self.value.bit_count()

    def ipv4(self):
        if self.version == 4:
            return IPAddress(self.value, 4)
        if self.value <= (1 << 32) - 1:
            return IPAddress(self.value, 4)
        if 0xFFFF00000000 <= self.value <= 0xFFFFFFFFFFFF:
            return IPAddress(self.value - 0xFFFF00000000, 4)
        raise AddrConversionError(f"IPv6 address {self} unsuitable for conversion to IPv4!")

    def ipv6(self, ipv4_compatible=False):
        if self.version == 6:
            if ipv4_compatible and 0xFFFF00000000 <= self.value <= 0xFFFFFFFFFFFF:
                return IPAddress(self.value - 0xFFFF00000000, 6)
            return IPAddress(self.value, 6)
        offset = 0 if ipv4_compatible else 0xFFFF00000000
        return IPAddress(offset + self.value, 6)

    def to_canonical(self):
        return self.ipv4() if self.is_ipv4_mapped() else self

    def is_ipv4_private_use(self):
        return self.version == 4 and any(
            self._within(cidr)
            for cidr in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")
        )

    def is_ipv6_unique_local(self):
        return self._within("fc00::/7")

    def is_global(self):
        if self.version == 4:
            blocked = (
                "0.0.0.0/8",
                "10.0.0.0/8",
                "100.64.0.0/10",
                "127.0.0.0/8",
                "169.254.0.0/16",
                "172.16.0.0/12",
                "192.0.0.0/24",
                "192.0.2.0/24",
                "192.168.0.0/16",
                "198.18.0.0/15",
                "198.51.100.0/24",
                "203.0.113.0/24",
                "240.0.0.0/4",
            )
            exceptions = ("192.0.0.9/32", "192.0.0.10/32")
        else:
            blocked = (
                "::1/128",
                "::/128",
                "::ffff:0:0/96",
                "64:ff9b:1::/48",
                "100::/64",
                "2001::/23",
                "2001:db8::/32",
                "2002::/16",
                "fc00::/7",
                "fe80::/10",
            )
            exceptions = (
                "2001:1::1/128",
                "2001:1::2/128",
                "2001:3::/32",
                "2001:4:112::/48",
                "2001:20::/28",
                "2001:30::/28",
            )
        return not any(self._within(cidr) for cidr in blocked) or any(
            self._within(cidr) for cidr in exceptions
        )


class IPListMixin:
    @property
    def size(self):
        return self.last - self.first + 1

    def __len__(self):
        if self.size > sys.maxsize:
            raise IndexError(
                f"range contains more than {sys.maxsize} (sys.maxsize) IP addresses! "
                "Use the .size property instead."
            )
        return self.size

    def __iter__(self):
        return iter_iprange(
            IPAddress(self.first, self.version), IPAddress(self.last, self.version)
        )

    def __getitem__(self, index):
        if isinstance(index, slice):
            if self.version == 6:
                raise TypeError("IPv6 slices are not supported!")
            start, stop, step = index.indices(self.size)
            return (
                IPAddress(self.first + offset, self.version)
                for offset in range(start, stop, step)
            )
        index = int(index)
        if index < 0:
            index += self.size
        if not 0 <= index < self.size:
            raise IndexError("index out range for address range size!")
        return IPAddress(self.first + index, self.version)

    def __bool__(self):
        return True


def _expand_partial(value: str) -> str:
    if ":" in value:
        return value
    pieces = value.split(".")
    if not 1 <= len(pieces) <= 4:
        return value
    return ".".join(pieces + ["0"] * (4 - len(pieces)))


class IPNetwork(BaseIP, IPListMixin):
    __slots__ = ("_prefixlen",)

    def __init__(self, addr, version=None, flags=0, *, expand_partial=False):
        if flags & ~NOHOST:
            raise ValueError(f"Unrecognized IPAddress flags value: {flags}")
        if version not in (None, 4, 6):
            raise ValueError(f"{version!r} is an invalid IP version!")
        if isinstance(addr, IPNetwork):
            self._value, self._version, self._prefixlen = (
                addr.value,
                addr.version,
                addr.prefixlen,
            )
        elif isinstance(addr, IPAddress):
            self._value, self._version = addr.value, addr.version
            self._prefixlen = _width(addr.version)
        elif isinstance(addr, tuple):
            if len(addr) != 2:
                raise AddrFormatError("invalid address tuple!")
            value, prefix = int(addr[0]), int(addr[1])
            detected = version or (4 if value <= (1 << 32) - 1 else 6)
            if not 0 <= value <= _maximum(detected):
                raise AddrFormatError("invalid address value for tuple!")
            self._value, self._version, self._prefixlen = value, detected, prefix
        elif isinstance(addr, str):
            base, slash, mask = addr.partition("/")
            if expand_partial:
                base = _expand_partial(base)
            value, detected = _parse_text(base, version, 0)
            width = _width(detected)
            if not slash:
                prefix = width
            else:
                try:
                    prefix = int(mask)
                except ValueError:
                    mask_ip = IPAddress(mask, detected)
                    if mask_ip.is_netmask():
                        prefix = mask_ip.netmask_bits()
                    elif mask_ip.is_hostmask():
                        prefix = width - mask_ip.value.bit_count()
                    else:
                        raise AddrFormatError(f"addr {addr!r} is not a valid IPNetwork!")
            self._value, self._version, self._prefixlen = value, detected, prefix
        else:
            raise TypeError(f"unexpected type {type(addr)} for addr arg")
        if not 0 <= self._prefixlen <= _width(self.version):
            raise AddrFormatError(f"invalid prefix for IPv{self.version} address!")
        if flags & NOHOST:
            self._value = self.first

    @property
    def prefixlen(self):
        return self._prefixlen

    @prefixlen.setter
    def prefixlen(self, value):
        if not isinstance(value, int):
            raise TypeError(f"int argument expected, not {type(value)}")
        if not 0 <= value <= _width(self.version):
            raise AddrFormatError(f"invalid prefix for an IPv{self.version} address!")
        self._prefixlen = value

    @property
    def ip(self):
        return IPAddress(self.value, self.version)

    @property
    def hostmask(self):
        return IPAddress((1 << (_width(self.version) - self.prefixlen)) - 1, self.version)

    @property
    def netmask(self):
        return IPAddress(_maximum(self.version) ^ int(self.hostmask), self.version)

    @netmask.setter
    def netmask(self, value):
        mask = IPAddress(value)
        if mask.version != self.version:
            raise ValueError(f"IP version mismatch: {mask} and {self}")
        if not mask.is_netmask():
            raise ValueError(f"Invalid subnet mask specified: {value}")
        self.prefixlen = mask.netmask_bits()

    @property
    def first(self):
        hostmask = (1 << (_width(self.version) - self.prefixlen)) - 1
        return self.value & (_maximum(self.version) ^ hostmask)

    @property
    def last(self):
        return self.value | ((1 << (_width(self.version) - self.prefixlen)) - 1)

    @property
    def network(self):
        return IPAddress(self.first, self.version)

    @property
    def broadcast(self):
        return None if _width(self.version) - self.prefixlen <= 1 else IPAddress(
            self.last, self.version
        )

    @property
    def cidr(self):
        return IPNetwork((self.first, self.prefixlen), self.version)

    def key(self):
        return self.version, self.first, self.last

    def sort_key(self):
        return self.version, self.first, self.prefixlen - 1, self.value - self.first

    def __contains__(self, other):
        if not isinstance(other, BaseIP):
            try:
                other = IPNetwork(other)
            except (AddrFormatError, TypeError, ValueError):
                return False
        if other.version != self.version:
            return False
        first, last = other._bounds()
        return self.first <= first and last <= self.last

    def __iadd__(self, num):
        value = self.first + self.size * num
        if value < 0 or value + self.size - 1 > _maximum(self.version):
            raise IndexError("increment exceeds address boundary!")
        self._value = value
        return self

    def __isub__(self, num):
        return self.__iadd__(-num)

    def previous(self, step=1):
        result = self.cidr
        result -= step
        return result

    def next(self, step=1):
        result = self.cidr
        result += step
        return result

    def supernet(self, prefixlen=0):
        if not 0 <= prefixlen <= _width(self.version):
            raise ValueError(f"CIDR prefix /{prefixlen} invalid for IPv{self.version}!")
        if prefixlen > self.prefixlen:
            return []
        return [
            IPNetwork((self.first, prefix), self.version).cidr
            for prefix in range(prefixlen, self.prefixlen)
        ]

    def subnet(self, prefixlen, count=None, fmt=None):
        if not self.prefixlen <= prefixlen <= _width(self.version):
            return
        total = 1 << (prefixlen - self.prefixlen)
        count = total if count is None else count
        if not 1 <= count <= total:
            raise ValueError("count outside of current IP subnet boundary!")
        size = 1 << (_width(self.version) - prefixlen)
        for i in range(count):
            yield IPNetwork((self.first + i * size, prefixlen), self.version)

    def iter_hosts(self):
        if self.size < 4:
            first, last = self.first, self.last
        else:
            first = self.first + 1
            last = self.last - 1 if self.version == 4 else self.last
        return iter_iprange(IPAddress(first, self.version), IPAddress(last, self.version))

    def ipv4(self):
        converted = self.ip.ipv4()
        return IPNetwork((converted.value, self.prefixlen - 96), 4)

    def ipv6(self, ipv4_compatible=False):
        converted = self.ip.ipv6(ipv4_compatible)
        prefix = self.prefixlen if self.version == 6 else self.prefixlen + 96
        return IPNetwork((converted.value, prefix), 6)

    def contains_many(self, addresses: Iterable) -> np.ndarray:
        values = []
        for item in addresses:
            ip = item if isinstance(item, IPAddress) else IPAddress(item, self.version)
            if ip.version != self.version:
                raise TypeError("address version does not match network")
            values.append(ip.value)
        return _lib.contains_many(self.first, self.prefixlen, _width(self.version), values)

    def __str__(self):
        return f"{self.ip}/{self.prefixlen}"

    def __repr__(self):
        return f"IPNetwork('{self}')"


class IPRange(BaseIP, IPListMixin):
    __slots__ = ("_start", "_end")

    def __init__(self, start, end, flags=0):
        self._start = IPAddress(start, flags=flags)
        self._end = IPAddress(end, self._start.version, flags=flags)
        self._version = self._start.version
        self._value = self._start.value
        if self._start.value > self._end.value:
            raise AddrFormatError("lower bound IP greater than upper bound!")

    @property
    def first(self):
        return self._start.value

    @property
    def last(self):
        return self._end.value

    def key(self):
        return self.version, self.first, self.last

    def sort_key(self):
        return self.version, self.first, _width(self.version) - self.size.bit_length()

    def __contains__(self, other):
        if not isinstance(other, BaseIP):
            other = IPAddress(other)
        if other.version != self.version:
            return False
        first, last = other._bounds()
        return self.first <= first and last <= self.last

    def cidrs(self):
        return iprange_to_cidrs(self._start, self._end)

    def __str__(self):
        return f"{self._start}-{self._end}"

    def __repr__(self):
        return f"IPRange('{self._start}', '{self._end}')"


def iter_iprange(start, end, step=1) -> Iterator[IPAddress]:
    start_ip, end_ip = IPAddress(start), IPAddress(end)
    if start_ip.version != end_ip.version:
        raise TypeError("start and stop IP versions do not match!")
    step = int(step)
    if step == 0:
        raise ValueError("step argument cannot be zero")
    stop = end_ip.value + (1 if step > 0 else -1)
    for value in range(start_ip.value, stop, step):
        yield IPAddress(value, start_ip.version)


def iprange_to_cidrs(start, end):
    start_net, end_net = IPNetwork(start), IPNetwork(end)
    if start_net.version != end_net.version:
        raise TypeError("IP sequence cannot contain both IPv4 and IPv6!")
    if start_net.first > end_net.last:
        raise AddrFormatError("lower bound IP greater than upper bound!")
    return _networks_from_tuples(
        _lib.range_to_tuples(
            start_net.first, end_net.last, _width(start_net.version)
        ),
        start_net.version,
    )


def _networks_from_tuples(items, version):
    result = []
    append = result.append
    for value, prefix in items:
        network = IPNetwork.__new__(IPNetwork)
        network._value = value
        network._version = version
        network._prefixlen = prefix
        append(network)
    return result


def cidr_merge(ip_addrs):
    if not hasattr(ip_addrs, "__iter__"):
        raise ValueError("A sequence or iterator is expected!")
    by_version: dict[int, list[tuple[int, int]]] = {4: [], 6: []}
    for item in ip_addrs:
        net = item if isinstance(item, (IPNetwork, IPRange)) else IPNetwork(item)
        version = net._version
        if isinstance(net, IPNetwork):
            width = 32 if version == 4 else 128
            if net._prefixlen == width:
                first = last = net._value
            else:
                hostmask = (1 << (width - net._prefixlen)) - 1
                first = net._value & (((1 << width) - 1) ^ hostmask)
                last = net._value | hostmask
        else:
            first, last = net._start._value, net._end._value
        by_version[version].append((first, last))
    result = []
    for version in (4, 6):
        ranges = by_version[version]
        ranges.sort()
        result.extend(
            _networks_from_tuples(
                _lib.merge_ranges(ranges, _width(version)), version
            )
        )
    return result


def cidr_exclude(target, exclude):
    target, exclude = IPNetwork(target).cidr, IPNetwork(exclude).cidr
    if target.version != exclude.version or exclude.last < target.first or target.last < exclude.first:
        return [target]
    if exclude.first <= target.first and target.last <= exclude.last:
        return []
    result = []
    if target.first < exclude.first:
        result.extend(
            iprange_to_cidrs(
                IPAddress(target.first, target.version),
                IPAddress(exclude.first - 1, target.version),
            )
        )
    if exclude.last < target.last:
        result.extend(
            iprange_to_cidrs(
                IPAddress(exclude.last + 1, target.version),
                IPAddress(target.last, target.version),
            )
        )
    return result


def spanning_cidr(ip_addrs):
    nets = [IPNetwork(item) for item in ip_addrs]
    if len(nets) < 2:
        raise ValueError("IP sequence must contain at least 2 elements!")
    versions = {net.version for net in nets}
    if len(versions) != 1:
        raise TypeError("IP sequence cannot contain both IPv4 and IPv6!")
    first, last = min(net.first for net in nets), max(net.last for net in nets)
    width = _width(nets[0].version)
    prefix = width - (first ^ last).bit_length()
    network = first & (_maximum(nets[0].version) ^ ((1 << (width - prefix)) - 1))
    return IPNetwork((network, prefix), nets[0].version)


def iter_unique_ips(*args):
    return chain.from_iterable(cidr_merge(args))


def all_matching_cidrs(ip, cidrs):
    address = IPAddress(ip)
    return [net for net in sorted(IPNetwork(cidr) for cidr in cidrs) if address in net]


def smallest_matching_cidr(ip, cidrs):
    matches = all_matching_cidrs(ip, cidrs)
    return max(matches, key=lambda net: net.prefixlen, default=None)


def largest_matching_cidr(ip, cidrs):
    matches = all_matching_cidrs(ip, cidrs)
    return min(matches, key=lambda net: net.prefixlen, default=None)


def cidr_abbrev_to_verbose(abbrev_cidr):
    def classful_prefix(octet):
        octet = int(octet)
        if not 0 <= octet <= 255:
            raise IndexError
        return 8 if octet <= 127 else 16 if octet <= 191 else 24 if octet <= 223 else 4 if octet <= 239 else 32

    if not isinstance(abbrev_cidr, str) or ":" in abbrev_cidr or not abbrev_cidr:
        return abbrev_cidr
    address, slash, prefix = abbrev_cidr.partition("/")
    pieces = address.split(".")
    if len(pieces) > 4:
        return abbrev_cidr
    try:
        pieces = [str(int(piece)) for piece in pieces]
        selected = int(prefix) if slash else classful_prefix(pieces[0])
        if not 0 <= selected <= 32 or any(not 0 <= int(piece) <= 255 for piece in pieces):
            return abbrev_cidr
    except (ValueError, IndexError):
        return abbrev_cidr
    return f"{'.'.join(pieces + ['0'] * (4 - len(pieces)))}/{selected}"


def expand_partial_ipv4_address(addr):
    result = _expand_partial(addr)
    IPAddress(result, 4)
    return result


def valid_ipv4(addr, flags=0):
    try:
        IPAddress(addr, 4, flags)
        return True
    except (AddrFormatError, TypeError, ValueError):
        return False


def valid_ipv6(addr):
    try:
        IPAddress(addr, 6)
        return True
    except (AddrFormatError, TypeError, ValueError):
        return False


class IPSet:
    __slots__ = ("_cidrs",)

    def __init__(self, iterable=None, flags=0):
        if isinstance(iterable, IPSet):
            self._cidrs = list(iterable._cidrs)
        elif iterable is None:
            self._cidrs = []
        elif isinstance(iterable, (IPNetwork, IPRange)):
            self._cidrs = cidr_merge([iterable])
        else:
            self._cidrs = cidr_merge(iterable)

    @property
    def size(self):
        return sum(cidr.size for cidr in self._cidrs)

    def __bool__(self):
        return bool(self._cidrs)

    def __hash__(self):
        raise TypeError("IP sets are unhashable!")

    def __iter__(self):
        return chain.from_iterable(sorted(self._cidrs))

    def __contains__(self, item):
        try:
            return any(item in cidr for cidr in self._cidrs)
        except (AddrFormatError, TypeError, ValueError):
            return False

    def __eq__(self, other):
        return isinstance(other, IPSet) and self._cidrs == other._cidrs

    def __le__(self, other):
        return self.issubset(other)

    def __lt__(self, other):
        return self.size < other.size and self.issubset(other)

    def __ge__(self, other):
        return self.issuperset(other)

    def __gt__(self, other):
        return self.size > other.size and self.issuperset(other)

    def __repr__(self):
        return f"IPSet({[str(cidr) for cidr in self._cidrs]!r})"

    def __str__(self):
        return f"IPSet({[str(cidr) for cidr in self._cidrs]!r})"

    def iter_cidrs(self):
        return iter(sorted(self._cidrs))

    def iter_ipranges(self):
        for version in (4, 6):
            nets = [net for net in sorted(self._cidrs) if net.version == version]
            if not nets:
                continue
            start, end = nets[0].first, nets[0].last
            for net in nets[1:]:
                if net.first == end + 1:
                    end = net.last
                else:
                    yield IPRange(IPAddress(start, version), IPAddress(end, version))
                    start, end = net.first, net.last
            yield IPRange(IPAddress(start, version), IPAddress(end, version))

    def copy(self):
        return IPSet(self)

    def clear(self):
        self._cidrs = []

    def compact(self):
        self._cidrs = cidr_merge(self._cidrs)

    def add(self, addr, flags=0):
        self._cidrs = cidr_merge(chain(self._cidrs, [addr]))

    def update(self, iterable, flags=0):
        items = iterable._cidrs if isinstance(iterable, IPSet) else (
            [iterable] if isinstance(iterable, (IPNetwork, IPRange)) else iterable
        )
        self._cidrs = cidr_merge(chain(self._cidrs, items))

    def remove(self, addr, flags=0):
        removal = addr if isinstance(addr, (IPNetwork, IPRange)) else IPNetwork(addr)
        remaining = []
        for cidr in self._cidrs:
            if cidr.version != removal.version or removal.last < cidr.first or cidr.last < removal.first:
                remaining.append(cidr)
                continue
            if cidr.first < removal.first:
                remaining.extend(
                    iprange_to_cidrs(
                        IPAddress(cidr.first, cidr.version),
                        IPAddress(removal.first - 1, cidr.version),
                    )
                )
            if removal.last < cidr.last:
                remaining.extend(
                    iprange_to_cidrs(
                        IPAddress(removal.last + 1, cidr.version),
                        IPAddress(cidr.last, cidr.version),
                    )
                )
        self._cidrs = cidr_merge(remaining)

    def pop(self):
        return self._cidrs.pop()

    def union(self, other):
        result = self.copy()
        result.update(other)
        return result

    __or__ = union

    def intersection(self, other):
        other = other if isinstance(other, IPSet) else IPSet(other)
        ranges = []
        for left in self._cidrs:
            for right in other._cidrs:
                if left.version == right.version:
                    start, end = max(left.first, right.first), min(left.last, right.last)
                    if start <= end:
                        ranges.append(
                            IPRange(
                                IPAddress(start, left.version),
                                IPAddress(end, left.version),
                            )
                        )
        return IPSet(ranges)

    __and__ = intersection

    def difference(self, other):
        result = self.copy()
        for cidr in (other._cidrs if isinstance(other, IPSet) else IPSet(other)._cidrs):
            result.remove(cidr)
        return result

    __sub__ = difference

    def symmetric_difference(self, other):
        other = other if isinstance(other, IPSet) else IPSet(other)
        return self.difference(other).union(other.difference(self))

    __xor__ = symmetric_difference

    def issubset(self, other):
        other = other if isinstance(other, IPSet) else IPSet(other)
        return all(cidr in other for cidr in self._cidrs)

    def issuperset(self, other):
        return (other if isinstance(other, IPSet) else IPSet(other)).issubset(self)

    def isdisjoint(self, other):
        return not self.intersection(other)

    def iscontiguous(self):
        return len(list(self.iter_ipranges())) == 1

    def iprange(self):
        ranges = list(self.iter_ipranges())
        if len(ranges) != 1:
            raise ValueError("IPSet is not contiguous")
        return ranges[0]
