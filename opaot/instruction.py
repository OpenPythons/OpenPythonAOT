from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, Enum
from functools import lru_cache
from typing import Union, Optional, Dict, Tuple, Callable


def signed(value: int, mask: int) -> int:
    if value & mask:
        value = (value ^ mask) - mask

    return value


class Op(Enum):
    MOVS = "MOVS"
    ADDS = "ADDS"
    SUBS = "SUBS"

    MOV = "MOV"
    ADD = "ADD"
    SUB = "SUB"

    ANDS = "ANDS"
    EORS = "EORS"
    LSLS = "LSLS"
    LSRS = "LSRS"
    ASRS = "ASRS"
    ADCS = "ADCS"
    SBCS = "SBCS"
    RORS = "RORS"
    TST = "TST"
    NEGS = "NEGS"
    CMP = "CMP"
    CMN = "CMN"
    ORRS = "ORRS"
    MULS = "MULS"
    BICS = "BICS"
    MVNS = "MVNS"

    LDR = "LDR"
    LDRB = "LDRB"
    LDRH = "LDRH"
    LDSB = "LDSB"
    LDSH = "LDSH"

    STR = "STR"
    STRB = "STRB"
    STRH = "STRH"

    PUSH = "PUSH"
    POP = "POP"

    STMIA = "STMIA"
    LDMIA = "LDMIA"

    B = "B"
    BL = "BL"
    BX = "BX"
    BLX = "BLX"
    BEQ = "BEQ"
    BNE = "BNE"
    BCS = "BCS"
    BCC = "BCC"
    BMI = "BMI"
    BPL = "BPL"
    BVS = "BVS"
    BVC = "BVC"
    BHI = "BHI"
    BLS = "BLS"
    BGE = "BGE"
    BLT = "BLT"
    BGT = "BGT"
    BLE = "BLE"
    SWI = "SWI"

    SXTH = "SXTH"
    SXTB = "SXTB"
    UXTH = "UXTH"
    UXTB = "UXTB"
    REV = "REV"

    def __str__(self):
        return self.name


class Reg(IntEnum):
    r0 = 0
    r1 = 1
    r2 = 2
    r3 = 3
    r4 = 4
    r5 = 5
    r6 = 6
    r7 = 7
    r8 = 8
    r9 = 9
    r10 = 10
    r11 = 11
    r12 = 12
    sp = 13
    lr = 14
    pc = 15

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"<{type(self).__name__}: {self.name}>"


class Imm(int):
    def __repr__(self):
        return f"<{type(self).__name__}: {int.__repr__(self)}>"


class Offset(int):
    def __repr__(self):
        return f"<{type(self).__name__}: {int.__repr__(self)}>"


@dataclass(frozen=True)
class Instruction:
    op: Op
    rd: Optional[Union[Reg, Imm, Offset]] = None
    rs: Optional[Union[Reg, Imm, Offset]] = None
    rn: Optional[Union[Reg, Imm, Offset]] = None
    flag: Optional[int] = None

    def __str__(self):
        args = []
        for name, value in ("rd", self.rd), ("rs", self.rs), ("rn", self.rn), ("flag", self.flag):
            if value is not None:
                args.append(f"{value!s}")

        return f"{self.op} {', '.join(args)}"


patterns: Dict[Pattern, Callable] = {}


@dataclass(init=False, frozen=True)
class Pattern:
    items: Tuple[str, ...]
    op: Optional[Op]
    cond: int
    mask: int
    args: Tuple[Tuple[str, int, int], ...]
    func: Optional[Callable] = field(hash=False, compare=False)

    def __init__(self, *items: str, op: Optional[Op] = None):
        super().__setattr__('items', items)
        super().__setattr__('op', op)
        cond, mask, args = self.build(self.items)
        super().__setattr__('cond', cond)
        super().__setattr__('mask', mask)
        super().__setattr__('args', args)
        super().__setattr__('func', None)

    def set_func(self, func: Callable):
        if self.func is not None:
            raise ValueError('Already function is set')

        super().__setattr__('func', func)

    def build(self, pattern: Tuple[str, ...]):
        pos = 16
        cond = 0
        mask = 0
        args = []
        for item in pattern:
            assert isinstance(item, str)
            if item.count('0') + item.count('1') == len(item):
                value = int(item, 2)
                item, size = None, len(item)
            else:
                item, sep, value = item.partition('=')
                value = int(value) if sep else None
                item, size = self._get_item_size(item)

            pos -= size

            if item is not None:
                args.append((item, pos, size))

            if value is not None:
                mask |= ((1 << size) - 1) << pos
                cond |= value << pos

        assert pos == 0, (pattern, pos)
        return cond, mask, tuple(args)

    @staticmethod
    def _get_item_size(item: str) -> Tuple[str, int]:
        for name, size in ('Rlist', 8), ('Cond', 4):
            if item == name:
                return item, size

        if item.isupper():
            return item, 1
        elif item.startswith('R'):
            return item, 3
        else:
            for name in 'Op', 'Offset', 'Word', 'SWord', 'SOffset', 'Value':
                if item.lower().startswith(name.lower()):
                    size = int(item[len(name):])
                    return f'{name}{size}', size

        raise Exception(item)

    def match(self, code: int) -> bool:
        return (code & self.mask) == self.cond

    def apply(self, code: int) -> Optional[Dict[str, int]]:
        if not self.match(code):
            return None

        def bits(offset: int, size: int) -> int:
            return code >> offset & ((1 << size) - 1)

        result = {}

        if self.op is not None:
            result['op'] = self.op

        for item, pos, size in self.args:
            value = bits(pos, size)
            if item.isupper():
                value = bool(value)
            elif item.startswith('R') and item != 'Rlist':
                value = Reg(value)

            result[item.lower()] = value

        return result

    def __str__(self):
        return '-'.join(self.items)


def pattern(*items: Tuple[str, ...], op: Optional[Op] = None):
    pattern = Pattern(*items, op=op)

    def pattern_wrapper(func: Callable):
        patterns[pattern] = func
        pattern.set_func(func)
        return func

    return pattern_wrapper


@pattern('000', '00', 'Offset5', 'Rs', 'Rd', op=Op.LSLS)
@pattern('000', '01', 'Offset5', 'Rs', 'Rd', op=Op.LSRS)
@pattern('000', '10', 'Offset5', 'Rs', 'Rd', op=Op.ASRS)
def op_1(op: Op, offset5: int, rs: Reg, rd: Reg) -> Instruction:
    """Move shifted register"""
    if offset5 == 0:
        if op == Op.LSLS:
            return Instruction(
                op=Op.MOVS,
                rd=rd,
                rs=rs,
            )
        else:
            offset5 = 32

    return Instruction(
        op=op,
        rd=rd,
        rs=rs,
        rn=Imm(offset5),
    )


@pattern('00011', 'I', '0', 'Offset3', 'Rs', 'Rd', op=Op.ADDS)
@pattern('00011', 'I', '1', 'Offset3', 'Rs', 'Rd', op=Op.SUBS)
def op_2(op: Op, i: bool, offset3: int, rs: Reg, rd: Reg) -> Instruction:
    """Add/subtract"""
    return Instruction(
        op=op,
        rd=rd,
        rs=rs,
        rn=(Imm if i else Reg)(offset3),
    )


@pattern('001', '00', 'Rd', 'Offset8', op=Op.MOVS)
@pattern('001', '01', 'Rd', 'Offset8', op=Op.CMP)
@pattern('001', '10', 'Rd', 'Offset8', op=Op.ADDS)
@pattern('001', '11', 'Rd', 'Offset8', op=Op.SUBS)
def op_3(op: Op, rd: Reg, offset8: int) -> Instruction:
    """Move/compare/add/subtract immediate"""
    return Instruction(
        op=op,
        rd=rd,
        rs=Imm(offset8),
    )


OP_4_TABLE = {
    0: Op.ANDS,
    1: Op.EORS,
    2: Op.LSLS,
    3: Op.LSRS,
    4: Op.ASRS,
    5: Op.ADCS,
    6: Op.SBCS,
    7: Op.RORS,
    8: Op.TST,
    9: Op.NEGS,
    10: Op.CMP,
    11: Op.CMN,
    12: Op.ORRS,
    13: Op.MULS,
    14: Op.BICS,
    15: Op.MVNS,
}


@pattern('010000', 'Op4', 'Rs', 'Rd')
def op_4(op4: int, rs: Reg, rd: Reg) -> Instruction:
    """ALU operations"""
    op = OP_4_TABLE[op4]
    if op == Op.MULS:
        return Instruction(op=op, rd=rd, rs=rs, rn=rd)
    else:
        return Instruction(op=op, rd=rd, rs=rs)


@pattern('010001', '00', 'H1', 'H2', 'Rs', 'Rd', op=Op.ADD)
@pattern('010001', '01', 'H1', 'H2', 'Rs', 'Rd', op=Op.CMP)
@pattern('010001', '10', 'H1', 'H2', 'Rs', 'Rd', op=Op.MOV)
@pattern('010001', '11', 'H1', 'H2', 'Rs', 'Rd', op=Op.BX or Op.BLX)
def op_5(op: Op, h1: bool, h2: bool, rs: Reg, rd: Reg) -> Instruction:
    """Hi register operations/branch exchange"""
    if op == Op.BX and h1:
        op = Op.BLX

    def hi(reg: Reg, flag: bool):
        return Reg(reg.value + 8) if flag else reg

    if op == Op.ADD:
        return Instruction(
            op=op,
            rd=hi(rd, h1),
            rs=hi(rs, h2),
            rn=hi(rd, h1) if hi(rs, h2) == Reg.sp else None,
        )
    if op == Op.CMP or op == Op.MOV:
        return Instruction(
            op=op,
            rd=hi(rd, h1),
            rs=hi(rs, h2),
        )
    elif op == Op.BX or op == Op.BLX:
        return Instruction(
            op,
            rd=hi(rs, h2),
        )
    else:
        raise AssertionError


@pattern('01001', 'Rd', 'Word8')
def op_6(rd: Reg, word8: int) -> Instruction:
    """PC-relative load"""
    return Instruction(
        op=Op.LDR,
        rd=rd,
        rs=Reg.pc,
        rn=Imm(word8 * 4),
    )


@pattern('0101', 'L=0', 'B=0', '0', 'Ro', 'Rb', 'Rd', op=Op.STR)
@pattern('0101', 'L=0', 'B=1', '0', 'Ro', 'Rb', 'Rd', op=Op.STRB)
@pattern('0101', 'L=1', 'B=0', '0', 'Ro', 'Rb', 'Rd', op=Op.LDR)
@pattern('0101', 'L=1', 'B=1', '0', 'Ro', 'Rb', 'Rd', op=Op.LDRB)
def op_7(op: Op, l: bool, b: bool, ro: Reg, rb: Reg, rd: Reg) -> Instruction:
    """Load/store with register"""
    return Instruction(op=op, rd=rd, rs=rb, rn=ro)


@pattern('0101', 'H=0', 'S=0', '1', 'Ro', 'Rb', 'Rd', op=Op.STRH)
@pattern('0101', 'H=0', 'S=1', '1', 'Ro', 'Rb', 'Rd', op=Op.LDSB)
@pattern('0101', 'H=1', 'S=0', '1', 'Ro', 'Rb', 'Rd', op=Op.LDRH)
@pattern('0101', 'H=1', 'S=1', '1', 'Ro', 'Rb', 'Rd', op=Op.LDSH)
def op_8(op: Op, h: bool, s: bool, ro: Reg, rb: Reg, rd: Reg) -> Instruction:
    """Load/store sign-extended byte/halfword"""
    return Instruction(op=op, rd=rd, rs=rb, rn=ro)


@pattern('011', 'B=0', 'L=0', 'Offset5', 'Rb', 'Rd', op=Op.STR)
@pattern('011', 'B=0', 'L=1', 'Offset5', 'Rb', 'Rd', op=Op.LDR)
@pattern('011', 'B=1', 'L=0', 'Offset5', 'Rb', 'Rd', op=Op.STRB)
@pattern('011', 'B=1', 'L=1', 'Offset5', 'Rb', 'Rd', op=Op.LDRB)
def op_9(op: Op, b: bool, l: bool, offset5: int, rb: Reg, rd: Reg) -> Instruction:
    """Load/store with immediate offset"""
    return Instruction(
        op=op,
        rd=rd,
        rs=rb,
        rn=Imm(offset5 << (0 if b else 2)) if offset5 else None,
    )


@pattern('1000', 'L=0', 'Offset5', 'Rb', 'Rd', op=Op.STRH)
@pattern('1000', 'L=1', 'Offset5', 'Rb', 'Rd', op=Op.LDRH)
def op_10(op: Op, l: bool, offset5: int, rb: Reg, rd: Reg) -> Instruction:
    """Load/store halfword"""
    return Instruction(
        op=op,
        rd=rd,
        rs=rb,
        rn=Imm(offset5 << 1) if offset5 else None,
    )


@pattern('1001', 'L=0', 'Rd', 'Word8', op=Op.STR)
@pattern('1001', 'L=1', 'Rd', 'Word8', op=Op.LDR)
def op_11(op: Op, l: bool, rd: Reg, word8: int) -> Instruction:
    """SP-relative load/store"""
    return Instruction(
        op=op,
        rd=rd,
        rs=Reg.sp,
        rn=Imm(word8 << 2) if word8 else None,
    )


@pattern('1010', 'SP', 'Rd', 'Word8')
def op_12(sp: bool, rd: Reg, word8: int) -> Instruction:
    """Load address"""
    return Instruction(
        op=Op.ADD,
        rd=rd,
        rs=Reg.sp if sp else Reg.pc,
        rn=Imm(word8 << 2),
    )


@pattern('10110000', 'S', 'SWord7')
def op_13(s: bool, sword7: int) -> Instruction:
    """Add offset to stack pointer"""
    return Instruction(
        op=Op.SUB if s else Op.ADD,
        rd=Reg.sp,
        rs=Imm(sword7 << 2),
    )


@pattern('1011', 'L=0', '10', 'R', 'Rlist', op=Op.PUSH)
@pattern('1011', 'L=1', '10', 'R', 'Rlist', op=Op.POP)
def op_14(op: Op, l: bool, r: bool, rlist: int) -> Instruction:
    """Push/pop registers"""
    return Instruction(
        op=op,
        rd=Imm(rlist),
        flag=r,
    )


@pattern('1100', 'L=0', 'Rb', 'Rlist', op=Op.STMIA)
@pattern('1100', 'L=1', 'Rb', 'Rlist', op=Op.LDMIA)
def op_15(op: Op, l: bool, rb: Reg, rlist: int) -> Instruction:
    """Multiple load/store"""
    return Instruction(
        op=op,
        rd=rb,
        rs=Imm(rlist),
    )


@pattern('1101', 'Cond=0', 'SOffset8', op=Op.BEQ)
@pattern('1101', 'Cond=1', 'SOffset8', op=Op.BNE)
@pattern('1101', 'Cond=2', 'SOffset8', op=Op.BCS)
@pattern('1101', 'Cond=3', 'SOffset8', op=Op.BCC)
@pattern('1101', 'Cond=4', 'SOffset8', op=Op.BMI)
@pattern('1101', 'Cond=5', 'SOffset8', op=Op.BPL)
@pattern('1101', 'Cond=6', 'SOffset8', op=Op.BVS)
@pattern('1101', 'Cond=7', 'SOffset8', op=Op.BVC)
@pattern('1101', 'Cond=8', 'SOffset8', op=Op.BHI)
@pattern('1101', 'Cond=9', 'SOffset8', op=Op.BLS)
@pattern('1101', 'Cond=10', 'SOffset8', op=Op.BGE)
@pattern('1101', 'Cond=11', 'SOffset8', op=Op.BLT)
@pattern('1101', 'Cond=12', 'SOffset8', op=Op.BGT)
@pattern('1101', 'Cond=13', 'SOffset8', op=Op.BLE)
def op_16(op: Op, cond: int, soffset8: int) -> Instruction:
    """Conditional branch"""
    return Instruction(
        op=op,
        rd=Offset(signed(soffset8 << 1, 0b1_0000_0000) + 4),
    )


@pattern('1101', '1111', 'Value8')
def op_17(value8: int) -> Instruction:
    """Software Interrupt"""
    return Instruction(
        op=Op.SWI,
        rd=Imm(value8),
    )


@pattern('11100', 'Offset11')
def op_18(offset11: int) -> Instruction:
    """Unconditional branch"""
    return Instruction(
        op=Op.B,
        rd=Offset(signed(offset11 << 1, 0b1000_0000_0000) + 4),
    )


@pattern('1111', 'H', 'Offset11')
def op_19(h: bool, offset11: int) -> Instruction:
    """Long branch with link"""
    return Instruction(
        op=Op.BL,
        rd=Offset(offset11),
        flag=h,
    )


OP_A_TABLE = {
    0: Op.SXTH,
    1: Op.SXTB,
    2: Op.UXTH,
    3: Op.UXTB,
}


@pattern('10110010', 'Op2', 'Rs', 'Rd')
def op_a(op2: int, rs: Reg, rd: Reg) -> Instruction:
    op = OP_A_TABLE[op2]
    return Instruction(op=op, rd=rd, rs=rs)


OP_B_TABLE = {
    0: Op.REV
}


@pattern('10111010', 'Op2', 'Rs', 'Rd')
def op_b(op2: int, rs: Reg, rd: Reg) -> Optional[Instruction]:
    op = OP_B_TABLE.get(op2)
    if op is None:
        return None

    return Instruction(op=op, rd=rd, rs=rs)


CACHE_OFFSET = 8


# noinspection PyShadowingNames
def build_cache():
    cache = dict.fromkeys(range(0x10000 >> CACHE_OFFSET))
    for pattern in sorted(patterns, key=lambda pattern: pattern.mask):
        for i in range(0x10000 >> CACHE_OFFSET):
            if pattern.match(i << CACHE_OFFSET):
                cache[i] = pattern

    return cache


cache = build_cache()


@lru_cache(0x10000)
def decode(code: int) -> Optional[Instruction]:
    pattern = cache[code >> CACHE_OFFSET]
    if pattern is None:
        return None

    data = pattern.apply(code)
    insn = pattern.func(**data)
    return insn


def get_bl_jump(insn1: Instruction, insn2: Optional[Instruction]) -> Optional[int]:
    if not insn2:
        return None
    if not (insn1.op == insn2.op == Op.BL):
        return None
    elif not (not insn1.flag and insn2.flag):
        return None
    else:
        offset = (insn1.rd << 12) | (insn2.rd << 1)
        return signed(offset, 0b01000000_00000000_00000000) + 4
