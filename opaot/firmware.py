from bisect import bisect_right
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Union

from fpvgcc import gccMemoryMap
from fpvgcc.fpv import process_map_file

from .symbol import Symbol, Function, Object


class SymbolLookupTable:
    def __init__(self, table: Dict[str, Symbol]):
        self.vmap = sorted(table.values(), key=lambda symbol: symbol.address)
        self.kmap = [symbol.address for symbol in self.vmap]

    def __getitem__(self, address: int) -> Symbol:
        index = bisect_right(self.kmap, address) - 1
        symbol = self.vmap[index]
        if symbol.address <= address < symbol.end:
            return symbol

        raise KeyError(address)


@dataclass(frozen=True)
class Buffer:
    address: int
    size: int
    content: bytes = field(repr=False)

    @classmethod
    def load(cls, address: int, path: Path):
        content = path.read_bytes()

        return cls(
            address=address,
            size=len(content),
            content=content,
        )


class Firmware:
    firmware: Buffer
    symbols: Dict[str, Symbol]
    functions: Dict[str, Function]
    objects: Dict[str, Object]

    ADDRESS = 0x08000000

    def __init__(self, folder: Path):
        self.folder = folder
        self.buffer = Buffer.load(self.ADDRESS, folder / "firmware.bin")
        self.symbols = {}
        self.symbol_by_name: Dict[str, Symbol] = {}
        self.symbol_by_address: Dict[int, Symbol] = {}
        self.functions = {}
        self.objects = {}

        self.parse_map(folder / "firmware.map")
        self.parse_gcc_map(folder / "firmware.elf.map")

        self.symbol_lookup_table = SymbolLookupTable(self.symbols)

    def __getstate__(self):
        return vars(self)

    def __setstate__(self, state):
        vars(self).update(state)

    def __getitem__(self, item: Union[str, int]) -> Union[Symbol, Function, Object]:
        if isinstance(item, int):
            return self.symbol_by_address[item]
        elif isinstance(item, str):
            return self.symbol_by_name[item]
        else:
            raise TypeError

    def parse_map(self, firmware_map_path: Path):
        begin = self.buffer.address
        content = self.buffer.content

        with firmware_map_path.open('r') as fp:
            for line in fp:
                address, size, tag, name = line.rstrip('\r\n').split('\t')
                address = int(address, 16)
                size = int(size, 16)

                if tag == "FUNC":
                    assert address & 1, "THUMB"
                    address &= ~1

                buf = content[address - begin:address - begin + size]

                if tag == "FUNC":
                    self.add(Function(name, address, size, address + size, buf))
                elif tag == "OBJECT":
                    self.add(Object(name, address, size, address + size, buf))

    def parse_gcc_map(self, firmware_gcc_map_path: Path):
        begin = self.buffer.address
        content = self.buffer.content

        ignore_regions = {"DISCARDED", "UNDEF"}

        sm = process_map_file(firmware_gcc_map_path)

        def visit(node: gccMemoryMap.GCCMemoryMapNode, gident: str):
            for node in node.children:  # type: gccMemoryMap.GCCMemoryMapNode
                visit(node, f'{gident}.{node.name}')

            if node.region not in ignore_regions:
                address, size = node._address, node.size
                for tag, prefix in ('OBJECT', '.text.rodata.'), ('FUNC', '.text.'):
                    if gident.startswith(prefix):
                        name = gident[len(prefix):]
                        break
                else:
                    return

                if tag == "FUNC" and name == "rodata":
                    return

                buf = content[address - begin:address - begin + size]

                if name not in self.symbols:
                    if tag == "FUNC":
                        self.add(Function(name, address, size, address + size, buf))
                    elif tag == "OBJECT":
                        self.add(Object(name, address, size, address + size, buf))

        for node in sm.memory_map.root.children:
            visit(node, node.gident)

    def add(self, symbol: Union[Symbol, Function, Object]):
        self.symbols[symbol.name] = symbol
        self.symbol_by_address[symbol.address] = symbol
        self.symbol_by_name[symbol.name] = symbol

        if isinstance(symbol, Function):
            self.functions[symbol.name] = symbol
        elif isinstance(symbol, Object):
            self.objects[symbol.name] = symbol

    @lru_cache()
    def lookup(self, address: int) -> Union[Symbol, Function, Object]:
        symbol = self.symbol_by_address.get(address)
        if symbol is not None:
            return symbol

        return self.symbol_lookup_table[address]

    def __repr__(self):
        return f"<{type(self).__name__}: {self.folder.as_posix()!r}>"
