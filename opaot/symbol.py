from array import array
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Symbol:
    name: str
    address: int
    size: int
    end: int
    content: bytes = field(repr=False)

    def as_address(self, offset: int) -> int:
        assert 0 < offset < self.size, (self, offset)
        address = self.address + offset
        return address

    def as_offset(self, address: int) -> int:
        offset = address - self.address
        assert 0 < offset < self.size, (self, offset)
        return offset


@dataclass(frozen=True)
class Function(Symbol):
    codes: array = field(repr=False, hash=False, default_factory=lambda: array('I'))

    def __post_init__(self):
        content = self.content
        codes = self.codes

        for i in range(0, self.size, 2):
            code = int.from_bytes(content[i: i + 2], "little")
            codes.append(code)


@dataclass(frozen=True)
class Object(Symbol):
    pass
