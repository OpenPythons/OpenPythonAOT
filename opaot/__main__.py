from datetime import timedelta
from pathlib import Path

from cachier import cachier
from cachier import pickle_core

from opaot.firmware import Firmware
from opaot.patch_fpv import patch_fpv
from opaot.visitor import Visitor

pickle_core.EXPANDED_CACHIER_DIR = "../cache"


@cachier(stale_after=timedelta(hours=1))
def load_firmware() -> Firmware:
    return Firmware(Path(r"..\..\OpenPythonFirmware\firmwares\v1.1.0").resolve())


def main():
    patch_fpv()

    firmware = load_firmware()
    visitor = Visitor(firmware)
    visitor.visit_all()

    for function, info in visitor.functions.items():
        jumps = info.jumps
        eof = info.eof

        if not function.size:
            print(function.name, 'missing')
            continue

        print(function.name)
        for pos, insn in zip(range(max(eof) + 1), info.insns):
            if pos in jumps:
                print(">- JUMP -<")

            print(insn)

            if pos in eof:
                print("<= FIN =>")

        print()


if __name__ == '__main__':
    main()
