# ydb's commands, loaded into gdb by `tools/ydb/ydb` (`gdb -x ydb_gdb.py`).
#
# They read what gdb and the compiler already say - the debug information, the
# embedded extension's program facts, `Y --emit-ptx --lineinfo` through
# `ymap.py` - and decide nothing on their own. `help break`, `help locals` ...
# describe each.
#
# The launcher passes the program's source and the compiler in the
# environment: YDB_PROGRAM (the .ysu), YDB_Y (the compiler), YDB_CWD (the
# directory to compile in, for its .ysu_hw_profile).

import math
import os
import re
import struct
import sys

import gdb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ymap  # noqa: E402


def _program():
    return os.environ.get("YDB_PROGRAM")


def _sources():
    """The .ysu files the program's debug information names (basename -> path)."""
    out = {}
    try:
        text = gdb.execute("info sources", to_string=True)
    except gdb.error:
        return out
    for tok in re.split(r"[,\s]+", text):
        if tok.endswith(".ysu"):
            out.setdefault(os.path.basename(tok), tok)
    prog = _program()
    if prog:
        out.setdefault(os.path.basename(prog), prog)
    return out


def _function_file(name):
    """The file the function `name` is defined in, or None."""
    for lookup in (gdb.lookup_global_symbol, gdb.lookup_static_symbol):
        try:
            sym = lookup(name)
        except gdb.error:
            sym = None
        if sym is not None and sym.symtab is not None and sym.type is not None \
                and sym.type.code == gdb.TYPE_CODE_FUNC:
            return sym.symtab.filename
    return None


def translate_location(spec):
    """A Y location as a gdb linespec. NAME:LINE is the file NAME.ysu, or the
    file the function NAME is in; anything else is gdb's own syntax."""
    m = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*):(\d+)", spec.strip())
    if not m:
        return spec
    name, line = m.group(1), m.group(2)
    sources = _sources()
    if name + ".ysu" in sources:
        return "%s.ysu:%s" % (name, line)
    f = _function_file(name)
    if f:
        return "%s:%s" % (os.path.basename(f), line)
    return spec


class YBreak(gdb.Command):
    """Set a breakpoint at a Y location.

    break FILE:LINE    break kernel.ysu:42
    break NAME:LINE    NAME is a file's stem (kernel:42 is kernel.ysu:42) or a
                       function (scale:14 is line 14 of the file `scale` is in)
    break LINE         a line of the current file
    break FUNCTION     a function's first line
    ... if COND        stop only when COND holds

    With no argument, the current line."""

    def __init__(self):
        super().__init__("break", gdb.COMMAND_BREAKPOINTS, gdb.COMPLETE_LOCATION)

    def invoke(self, arg, from_tty):
        arg = arg.strip()
        cond = None
        m = re.match(r"^(.*?)\s+if\s+(.*)$", arg)
        if m:
            arg, cond = m.group(1).strip(), m.group(2).strip()
        if not arg:
            sal = gdb.selected_frame().find_sal()
            if sal.symtab is None:
                raise gdb.GdbError("no current source line to break at")
            arg = "%s:%d" % (sal.symtab.filename, sal.line)
        bp = gdb.Breakpoint(translate_location(arg))
        if cond:
            bp.condition = cond


def y_type_name(t):
    """A gdb type as Y spells it."""
    s = t.strip_typedefs() if t.code == gdb.TYPE_CODE_TYPEDEF and not t.name else t
    if s.code == gdb.TYPE_CODE_ARRAY:
        lo, hi = s.range()
        return "[%s; %d]" % (y_type_name(s.target()), hi - lo + 1)
    if s.code == gdb.TYPE_CODE_PTR:
        return "*" + y_type_name(s.target())
    # A struct or enum by its Y name, not gdb's C spelling (`struct Point`).
    return t.name or str(t)


def _visible_symbols(frame):
    """(argument?, symbol) for every argument and local in scope at the frame's
    pc, innermost binding first; an outer binding a `let` shadows is hidden, as
    it is in the program."""
    seen = set()
    out = []
    try:
        block = frame.block()
    except RuntimeError:
        return out
    while block is not None:
        for sym in block:
            if not (sym.is_variable or sym.is_argument) or sym.name in seen:
                continue
            seen.add(sym.name)
            out.append((sym.is_argument, sym))
        if block.function is not None:
            break
        block = block.superblock
    return out


class YLocals(gdb.Command):
    """The arguments and locals in scope, with their Y types:

        name: Type = value

    A binding shadowed by an inner `let` is not shown; it is not in scope."""

    def __init__(self):
        super().__init__("locals", gdb.COMMAND_DATA)

    def invoke(self, arg, from_tty):
        frame = gdb.selected_frame()
        syms = _visible_symbols(frame)
        if not syms:
            print("no locals in scope")
            return
        for is_arg, sym in sorted(syms, key=lambda s: (not s[0],)):
            try:
                v = frame.read_var(sym)
                text = "<optimized out>" if v.is_optimized_out else v.format_string()
            except gdb.error as e:
                text = "<%s>" % e
            print("%s%s: %s = %s" % ("(arg) " if is_arg else "", sym.name, y_type_name(sym.type), text))


_UNPACK = {
    (gdb.TYPE_CODE_INT, 1, True): "b", (gdb.TYPE_CODE_INT, 1, False): "B",
    (gdb.TYPE_CODE_INT, 2, True): "h", (gdb.TYPE_CODE_INT, 2, False): "H",
    (gdb.TYPE_CODE_INT, 4, True): "i", (gdb.TYPE_CODE_INT, 4, False): "I",
    (gdb.TYPE_CODE_INT, 8, True): "q", (gdb.TYPE_CODE_INT, 8, False): "Q",
    (gdb.TYPE_CODE_FLT, 4, True): "f", (gdb.TYPE_CODE_FLT, 8, True): "d",
    (gdb.TYPE_CODE_BOOL, 1, False): "?",
}


def _elements(expr, count):
    """(element type, address, n, list of values) for an array or a pointer."""
    v = gdb.parse_and_eval(expr)
    t = v.type.strip_typedefs()
    if t.code == gdb.TYPE_CODE_ARRAY:
        lo, hi = t.range()
        n = hi - lo + 1
        if count is not None:
            n = min(n, count)
        elem = t.target()
        addr = int(v.address) if v.address is not None else None
    elif t.code == gdb.TYPE_CODE_PTR:
        if count is None:
            raise gdb.GdbError("%s is a pointer, so its length is not known here: tensor %s COUNT" % (expr, expr))
        n = count
        elem = t.target()
        addr = int(v)
    else:
        raise gdb.GdbError("%s is a %s, not an array or a buffer" % (expr, y_type_name(v.type)))
    es = elem.strip_typedefs()
    if es.code in (gdb.TYPE_CODE_INT, gdb.TYPE_CODE_CHAR):
        key = (gdb.TYPE_CODE_INT, es.sizeof, bool(es.is_signed))
    elif es.code == gdb.TYPE_CODE_FLT:
        key = (gdb.TYPE_CODE_FLT, es.sizeof, True)
    elif es.code == gdb.TYPE_CODE_BOOL:
        key = (gdb.TYPE_CODE_BOOL, es.sizeof, False)
    else:
        key = None
    fmt = _UNPACK.get(key) if key else None
    if fmt is None or addr is None:
        # Not a plain number: read element by element through gdb.
        vals = []
        base = v if t.code == gdb.TYPE_CODE_ARRAY else None
        for i in range(n):
            vals.append(base[i] if base is not None else (v + i).dereference())
        return elem, addr, n, vals, False
    data = gdb.selected_inferior().read_memory(addr, n * es.sizeof).tobytes()
    vals = list(struct.unpack("<%d%s" % (n, fmt), data))
    return elem, addr, n, vals, True


def _fmt(x):
    if isinstance(x, float):
        return repr(x) if not math.isfinite(x) else ("%.9g" % x)
    return str(x)


class YTensor(gdb.Command):
    """A buffer's elements and summary statistics.

    tensor EXPR          an array: its length is known
    tensor EXPR COUNT    a pointer (a GlobalMemory parameter): COUNT elements
    """

    def __init__(self):
        super().__init__("tensor", gdb.COMMAND_DATA)

    def invoke(self, arg, from_tty):
        parts = arg.split()
        if not parts:
            raise gdb.GdbError("tensor EXPR [COUNT]")
        count = None
        if len(parts) > 1 and re.fullmatch(r"\d+", parts[-1]):
            count = int(parts[-1])
            parts = parts[:-1]
        expr = " ".join(parts)
        elem, addr, n, vals, numeric = _elements(expr, count)
        where = " at %#x" % addr if addr is not None else ""
        print("%s: %d x %s%s" % (expr, n, y_type_name(elem), where))
        shown = list(range(n)) if n <= 16 else list(range(8)) + [None] + list(range(n - 4, n))
        row = []
        for i in shown:
            if i is None:
                row.append("...")
                continue
            x = vals[i]
            row.append("[%d] %s" % (i, _fmt(x) if numeric else x.format_string()))
        for k in range(0, len(row), 8):
            print("  " + "  ".join(row[k:k + 8]))
        if not numeric or n == 0:
            return
        finite = [x for x in vals if not (isinstance(x, float) and not math.isfinite(x))]
        stats = []
        if finite:
            stats.append("min %s" % _fmt(min(finite)))
            stats.append("max %s" % _fmt(max(finite)))
            stats.append("mean %s" % _fmt(sum(float(x) for x in finite) / len(finite)))
        stats.append("zeros %d" % sum(1 for x in vals if x == 0))
        if isinstance(vals[0], float):
            stats.append("NaN %d" % sum(1 for x in vals if x != x))
            stats.append("inf %d" % sum(1 for x in vals if isinstance(x, float) and math.isinf(x)))
        print("  " + ", ".join(stats))


_MAP = {}


def _line_map():
    """The program's PTX/SASS line map, built once per session."""
    prog = _program()
    if not prog:
        raise gdb.GdbError("ydb did not say which source this program came from (YDB_PROGRAM)")
    if prog not in _MAP:
        try:
            _MAP[prog] = ymap.build_map(prog, y=os.environ.get("YDB_Y"), cwd=os.environ.get("YDB_CWD"))
        except ymap.ToolError as e:
            raise gdb.GdbError(str(e))
    return _MAP[prog]


def _current_line():
    sal = gdb.selected_frame().find_sal()
    if sal.symtab is None:
        raise gdb.GdbError("no current source line")
    return sal.symtab.filename, sal.line


def _host_ranges(filename, line):
    """[lo, hi) pc ranges of the code `line` of `filename` became."""
    symtab = None
    sal = gdb.selected_frame().find_sal()
    if sal.symtab is not None and os.path.basename(sal.symtab.filename) == os.path.basename(filename):
        symtab = sal.symtab
    if symtab is None:
        raise gdb.GdbError("asm LINE for the host reads the current file's line table; stop in %s first" % filename)
    entries = sorted(symtab.linetable(), key=lambda e: e.pc)
    ranges = []
    for i, e in enumerate(entries):
        if e.line != line:
            continue
        hi = next((x.pc for x in entries[i + 1:] if x.pc > e.pc), None)
        if hi is not None:
            ranges.append((e.pc, hi))
    return ranges


class YAsm(gdb.Command):
    """The code a Y line became.

    asm [LINE]           this program's machine code for the current line (or LINE)
    asm --ptx [LINE]     the PTX a kernel's line became (Y --emit-ptx --lineinfo)
    asm --sass [LINE]    the SASS ptxas made of it, at the module's own .target

    The PTX and SASS are the program compiled for the GPU, not the process
    being debugged: a kernel runs on the host here only when it uses no GPU
    intrinsic."""

    def __init__(self):
        super().__init__("asm", gdb.COMMAND_DATA)

    def invoke(self, arg, from_tty):
        parts = arg.split()
        mode = "host"
        for flag in ("--ptx", "--sass", "--host"):
            if flag in parts:
                mode = flag[2:]
                parts.remove(flag)
        if mode == "host" or not parts:
            filename, line = _current_line() if not parts else (_current_line()[0], None)
        else:
            filename = _program()
        if parts:
            if not re.fullmatch(r"\d+", parts[0]):
                raise gdb.GdbError("asm [--ptx|--sass] [LINE]")
            line = int(parts[0])
        if mode == "host":
            ranges = _host_ranges(filename, line)
            if not ranges:
                print("%s:%d has no machine code in this build" % (os.path.basename(filename), line))
                return
            for lo, hi in ranges:
                gdb.execute("disassemble %#x,%#x" % (lo, hi))
            return
        m = _line_map()
        if mode == "sass" and not m.get("sass"):
            raise gdb.GdbError("no SASS: %s" % (m.get("note") or "ptxas or nvdisasm unavailable"))
        rows = []
        for k in m["kernels"]:
            for r in k["lines"]:
                if os.path.basename(r["file"]) == os.path.basename(filename) and r["line"] == line:
                    rows.append((k["name"], r))
        if not rows:
            print("%s:%d is not in a kernel's code: it becomes no PTX" % (os.path.basename(filename), line))
            return
        for name, r in rows:
            insns = r["ptx"] if mode == "ptx" else r.get("sass", [])
            print("kernel %s  (%s)  %s:%d  %s" % (name, m["target"], os.path.basename(r["file"]), r["line"], r["source"]))
            if not insns:
                print("  %s: none - ptxas removed this line's code, or folded it into another line" % mode.upper()
                      if mode == "sass" and r["ptx"] else "  none")
            for i in insns:
                print("  " + i)


YBreak()
YLocals()
YTensor()
YAsm()
