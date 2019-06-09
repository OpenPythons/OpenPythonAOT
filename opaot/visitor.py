from dataclasses import dataclass, field
from typing import Optional, Tuple, Set, Dict

from .firmware import Firmware
from .instruction import Instruction, Op, Reg, Offset, get_bl_jump, decode
from .symbol import Function, Symbol

INSN_NOP = Instruction(op=Op.MOV, rd=Reg.r8, rs=Reg.r8)
COND_JUMPS = frozenset({
    Op.BEQ,
    Op.BNE,
    Op.BCS,
    Op.BCC,
    Op.BMI,
    Op.BPL,
    Op.BVS,
    Op.BVC,
    Op.BHI,
    Op.BLS,
    Op.BGE,
    Op.BLT,
    Op.BGT,
    Op.BLE,
})

OFFSET_JUMPS = frozenset({
    Op.B,
    Op.BX,
    Op.BL,
    Op.BLX,
})

JUMPS = frozenset(COND_JUMPS | OFFSET_JUMPS)


def is_jump(insn: Instruction):
    return insn.op in JUMPS or (insn.op == Op.POP and insn.flag)


TAIL_FUNCTIONS = {
    "__fatal_error",
    "_start",
    "nlr_jump",
    "nlr_jump_fail",
    "m_malloc_fail",
    "mp_hal_raise",
    "mp_arg_error_terse_mismatch",
    "mp_arg_error_unimpl_kw",
}


def is_tail_func(func: Function):
    return func.name in TAIL_FUNCTIONS or func.name.startswith("mp_raise_") or func.name.startswith("unlikely.")


@dataclass
class FunctionInfo:
    function: Function
    insns: Tuple[Optional[Instruction]] = field(default=())
    jumps: Set[int] = field(default_factory=set)
    visited: Set[int] = field(default_factory=set)
    eof: Set[int] = field(default_factory=set)

    def __post_init__(self):
        insns = [*map(decode, self.function.codes), None]
        for pos, insn in enumerate(insns):
            if insn is None:
                continue
            elif insn.op == Op.BL:
                next_insn = insns[pos + 1]
                offset = get_bl_jump(insn, next_insn)
                if offset is not None:
                    insns[pos] = Instruction(Op.BL, Offset(offset), flag=None)
                    insns[pos + 1] = None

        self.insns = tuple(insns)


class Visitor:
    functions: Dict[Function, FunctionInfo]

    def __init__(self, firmware: Firmware):
        self.firmware = firmware
        self.functions = {
            function: FunctionInfo(function)
            for function in self.firmware.functions.values()
        }

    def visit_all(self):
        self.visit_entry_table()
        for function in self.firmware.functions.values():
            self.visit(function)

    def visit(self, function: Function, pc: Optional[int] = None):
        info = self.functions[function]
        begin, end = function.address, function.end
        insns = info.insns
        visited = info.visited
        jumps = info.jumps
        eof = info.eof

        def visit(pc: int):
            pos = (pc - begin) >> 1
            if pos < 0:
                return

            jumps.add(pos)

            prev_insn = None
            while pc < end:
                if pos not in visited:
                    visited.add(pos)
                else:
                    return

                insn = insns[pos]
                if insn is None:
                    raise ValueError(insn)

                # print(function.name, hex(pc), pos, insn)
                if prev_insn is None and insn == INSN_NOP and function.name == "mp_execute_bytecode":
                    eof.add(pos - 1)
                    # crash detected
                    return

                if is_jump(insn):
                    new_pc, next_pc = self.next_op(pc, insn, insns[pos + 1])

                    if new_pc is not None:
                        if begin <= new_pc < end:
                            visit(new_pc)
                        else:
                            new_function = self.firmware.lookup(new_pc)
                            self.visit(new_function, new_pc)

                    if next_pc is not None:
                        prev_insn = None
                        pos += (next_pc - pc) >> 1
                        pc = next_pc
                    else:
                        eof.add(pos)
                        return
                else:
                    prev_insn = insn
                    pos += 1
                    pc += 2
            else:
                if function.size > 0:
                    assert False, (hex(pc), function)

        visit(begin if pc is None else pc)

    def next_op(self,
                pc: int,
                insn: Instruction,
                next_insn: Optional[Instruction] = None) -> Tuple[Optional[int], Optional[int]]:
        if insn.op == Op.POP and insn.flag:
            return None, None
        elif insn.op == Op.B:
            assert isinstance(insn.rd, Offset)
            new_pc = pc + int(insn.rd)
            next_pc = None
        elif insn.op == Op.BL:
            if insn.flag is None:
                offset = int(insn.rd)
            else:
                offset = get_bl_jump(insn, next_insn)

            if offset is not None:
                new_pc = pc + offset
                next_pc = pc + 4
            else:
                raise AssertionError((pc, insn))
        elif insn.op == Op.BX:
            assert isinstance(insn.rd, Reg)
            new_pc = None
            next_pc = None
        elif insn.op == Op.BLX:
            assert isinstance(insn.rd, Reg)
            new_pc = None
            next_pc = pc + 2
        elif insn.op in COND_JUMPS:
            assert isinstance(insn.rd, Offset)
            new_pc = pc + int(insn.rd)
            next_pc = pc + 2
        else:
            raise ValueError(insn)

        if new_pc is not None:
            new_symbol = self.firmware.lookup(new_pc)
            if is_tail_func(new_symbol):
                next_pc = None

        return new_pc, next_pc

    def visit_entry_table(self):
        mp_execute_bytecode = self.firmware.functions["mp_execute_bytecode"]
        entry_table = self._find_entry_table()

        for offset in range(0, 256 * 4, 4):
            pc = int.from_bytes(entry_table.content[offset: offset + 4], byteorder='little')
            assert mp_execute_bytecode.address <= pc < mp_execute_bytecode.end, pc
            self.visit(mp_execute_bytecode, pc)

    def _find_entry_table(self) -> Symbol:
        entry_table = None

        for name in self.firmware.symbols:
            if name.startswith('entry_table.'):
                assert entry_table is None
                entry_table = self.firmware.symbols[name]
        else:
            assert entry_table

        return entry_table
