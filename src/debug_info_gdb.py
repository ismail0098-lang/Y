# Y's extension for gdb: Y values printed as Y, and Y stacks shown as Y.
#
# The Y compiler embeds this file in every `-g` build, in the program's
# `.debug_gdb_scripts` section, so gdb loads it together with the program it
# describes. gdb runs it only for a program in its auto-load safe path:
# `Y prog.ysu --debug` adds the program it built; with plain gdb, gdb's own
# warning names the line to add (`add-auto-load-safe-path <program>`).
#
# It FORMATS what the debug information already says and decides nothing on
# its own: every printer below reads a value through the type the compiler
# described, so what it prints is that value. `print/r` bypasses it.
#
# The compiler prepends one line, `Y_PROGRAM = {...}`, describing the
# program: its enums, the symbols that are methods, and what the compiler
# checked, proved or assumed about each line (`guarantees`, which
# `tools/ydb`'s `verify` reads; `src/guarantees.rs`).

import re

import gdb
from gdb.FrameDecorator import FrameDecorator

try:
    Y_PROGRAM
except NameError:  # loaded by hand, outside a Y program
    Y_PROGRAM = {"enums": {}, "methods": {}, "guarantees": None}

_OBJFILE = gdb.current_objfile()


def _exact_decimal(raw, frac):
    """`raw / 2**frac` written out exactly: a dyadic fraction always ends."""
    sign = "-" if raw < 0 else ""
    raw = abs(raw)
    whole = raw >> frac
    rest = raw & ((1 << frac) - 1)
    if rest == 0:
        return "%s%d" % (sign, whole)
    # rest / 2**frac == rest * 5**frac / 10**frac
    digits = str(rest * 5 ** frac).rjust(frac, "0").rstrip("0")
    return "%s%d.%s" % (sign, whole, digits)


class StringPrinter:
    """A Y `String` - the runtime's `YStr*` - shown as its text."""

    def __init__(self, val):
        self.val = val
        self.hint = None

    def to_string(self):
        addr = int(self.val)
        if addr == 0:
            return "<String: null>"
        try:
            s = self.val.dereference()
            n = int(s["len"])
            if n < 0 or n > (1 << 28):
                return "<String at %#x: length %d>" % (addr, n)
            text = s["data"].lazy_string(length=n)
        except gdb.MemoryError:
            return "<String at %#x: unreadable>" % addr
        self.hint = "string"
        return text

    def display_hint(self):
        return self.hint


class FixedPointPrinter:
    """A `@ZeroDrift` accumulator: its exact value, and the representation
    the compiler chose for it. The slot holds `value * 2**frac`."""

    def __init__(self, val, whole, frac):
        self.val = val
        self.repr = "Q%d.%d" % (whole, frac)
        self.frac = frac

    def to_string(self):
        return "%s (%s)" % (_exact_decimal(int(self.val), self.frac), self.repr)


class IntPrinter:
    """`I8` / `U8`: a number. gdb prints a one-byte integer as a character
    too, the way C's `char` reads; a Y `I8` is not a character."""

    def __init__(self, val):
        self.val = val

    def to_string(self):
        return str(int(self.val))


class EnumPrinter:
    """A Y enum value, as `Enum::Variant`."""

    def __init__(self, val, name):
        self.val = val
        self.name = name

    def to_string(self):
        v = int(self.val)
        for f in self.val.type.strip_typedefs().fields():
            if f.enumval == v:
                return "%s::%s" % (self.name, f.name)
        return "%s::<invalid %d>" % (self.name, v)


_FIXED = re.compile(r"^Q(\d+)\.(\d+)_raw$")


def y_lookup(val):
    t = val.type
    name = t.unqualified().name or ""
    if name == "String":
        return StringPrinter(val)
    m = _FIXED.match(name)
    if m:
        return FixedPointPrinter(val, int(m.group(1)), int(m.group(2)))
    if name in ("I8", "U8"):
        return IntPrinter(val)
    st = t.strip_typedefs()
    if st.code == gdb.TYPE_CODE_ENUM:
        enum = st.name or ""
        if enum in Y_PROGRAM["enums"]:
            return EnumPrinter(val, enum)
        # A data-carrying enum's tag type is `<Enum>::tag`.
        if enum.endswith("::tag") and enum[: -len("::tag")] in Y_PROGRAM["enums"]:
            return EnumPrinter(val, enum[: -len("::tag")])
    return None


# --- Stack traces ---


def _is_y(frame):
    sal = frame.find_sal()
    return sal.symtab is not None and sal.symtab.filename.endswith(".ysu")


class YFrame(FrameDecorator):
    """A frame of Y code: the method's Y name, and no return address."""

    def __init__(self, base):
        super().__init__(base)
        self.base = base

    def function(self):
        f = self.base.function()
        if isinstance(f, str):
            return Y_PROGRAM["methods"].get(f, f)
        return f

    def address(self):
        return None


class YFrames:
    """The frames of Y code and of what Y code called, innermost first; the
    runtime's own frames below `fn main` - the C `main` that set up the
    allocator and stack, and libc's start-up - are left out. `bt -no-filters`
    shows every frame."""

    def __init__(self, frames):
        self.frames = frames
        self.done = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.done:
            raise StopIteration
        base = next(self.frames)
        frame = base.inferior_frame()
        if not _is_y(frame):
            return base
        if frame.name() == "main":
            self.done = True
        return YFrame(base)


class YFrameFilter:
    def __init__(self):
        self.name = "ysu"
        self.priority = 100
        self.enabled = True

    def filter(self, frames):
        return YFrames(frames)


if _OBJFILE is not None:
    _OBJFILE.pretty_printers.append(y_lookup)
    _OBJFILE.frame_filters["ysu"] = YFrameFilter()
else:
    gdb.pretty_printers.append(y_lookup)
    gdb.frame_filters["ysu"] = YFrameFilter()
