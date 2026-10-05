"""Explicit launch and target domain shared by every validator entry point.

CUDA permits grid.x up to 2**31-1 blocks, so ctaid.x may reach 2**31-2.
The previous 2**24 restriction silently excluded legal launches. ISA/ABI
interpretations remain trusted models. The current proof mode licenses those
models only on sm_89; matching declarations do not license another target.
"""
import re
from pathlib import Path
import ptxsource

LICENSED_TARGETS = frozenset(('sm_89',))


def require_licensed_target(target):
    """The ISA/ABI and empirical identifications have one reviewed target.

    Assembly portability is a separate property. Until another architecture's
    model assumptions are licensed, it must not receive a VALIDATED verdict.
    """
    if target not in LICENSED_TARGETS:
        raise Exception(f'UNMODELLED unsupported architecture {target!r}: '
                        'current proof mode licenses sm_89 ISA/ABI assumptions only '
                        '(refusing, not guessing)')
    return target


def launch_preconditions(sym):
    from z3 import ULT, BitVecVal
    return [ULT(sym['tid_x'], BitVecVal(1024, 32)),
            ULT(sym['ctaid_x'], BitVecVal(0x7fffffff, 32))]


def require_matching_targets(ptx, sass):
    def target(path, side):
        text = ptxsource.strip_comments(Path(path).read_text())
        declarations = re.findall(r'(?m)^\s*\.target\s+([^\r\n]+)', text)
        if len(declarations) != 1 or not re.fullmatch(r'sm_[0-9]+[af]?', declarations[0].strip()):
            raise Exception(f'UNMODELLED {side} target declaration (refusing, not guessing)')
        return declarations[0].strip()
    p, s = target(ptx, 'PTX'), target(sass, 'SASS')
    if p != s:
        raise Exception(f'UNMODELLED target mismatch: PTX {p}, SASS {s} (refusing, not guessing)')
    return require_licensed_target(p)
