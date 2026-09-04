"""Recover a kernel's parameter list and its constant-bank layout from the PTX.

Hardcoding one kernel's parameter names is how a validator silently becomes a
validator of one kernel.  The bank offsets follow the driver ABI: parameters
start at 0x160 on sm_8x/9x, each aligned to its own size, in declaration order.
"""
import re
SIZE = {'.u64':8,'.s64':8,'.b64':8,'.f64':8,
        '.u32':4,'.s32':4,'.b32':4,'.f32':4,
        '.u16':2,'.s16':2,'.b16':2,'.f16':2,'.u8':1,'.s8':1,'.b8':1}
PARAM_BASE = 0x160
def parse(path):
    txt = open(path).read()
    m = re.search(r'\.entry\s+(\w+)\s*\(\s*(.*?)\s*\)', txt, re.S)
    if not m: raise Exception('no kernel parameter list found')
    body = m.group(2)
    params = []
    for line in body.split(','):
        mm = re.search(r'\.param\s+(\.\w+)\s+(\w+)', line)
        if mm: params.append((mm.group(2), mm.group(1)))
    off = PARAM_BASE; layout = {}
    for name, ty in params:
        sz = SIZE[ty]
        off = (off + sz - 1) // sz * sz
        layout[name] = (off, sz, ty)
        off += sz
    return params, layout
def cbank(layout):
    """constant-bank offset -> (param name, which 32-bit half)"""
    m = {}
    for name,(off,sz,ty) in layout.items():
        if sz == 8:
            m[off] = (name,'lo'); m[off+4] = (name,'hi')
        else:
            m[off] = (name,'lo')
    return m
