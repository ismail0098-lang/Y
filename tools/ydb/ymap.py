#!/usr/bin/env python3
"""Which PTX and SASS did each Y line become?

    python3 tools/ydb/ymap.py kernel.ysu                 # every line of every kernel
    python3 tools/ydb/ymap.py kernel.ysu --line 22       # one line
    python3 tools/ydb/ymap.py kernel.ysu --kernel saxpy  # one kernel
    python3 tools/ydb/ymap.py kernel.ysu --json          # for tools (ydb)

It asks the compiler, the assembler and the disassembler, and reads what they
wrote - nothing here models a lowering:

  1. `Y kernel.ysu --emit-ptx --lineinfo -o <scratch>/kernel.ptx`: the PTX
     backend writes a line table, a `.loc` before the instructions of every
     statement.
  2. `ptxas -lineinfo -arch=<the module's own .target>`: never this machine's
     card. A kernel is compiled for the target the compiler chose, and that is
     the SASS this shows (tests/ptx_portability.rs says why the two differ).
  3. `nvdisasm -g`: every SASS instruction under the Y line ptxas recorded.

`-lineinfo` changes no instruction ptxas emits (tests/ptx_line_info.rs checks
that), so the SASS shown is the SASS the kernel runs.

A Y line with PTX and no SASS is code ptxas removed, usually because nothing
it computes is stored. ptxas also moves instructions across lines, so a line's
SASS is not always contiguous. The function-end padding ptxas emits after the
last EXIT (a self-branch and NOPs) belongs to no line and is not shown.

The compiler runs in the current directory, so it reads the same
`.ysu_hw_profile` (the target GPU) a plain `Y kernel.ysu --emit-ptx` there
would.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))


def find_y(explicit=None):
    """The Y compiler: --y, then $Y_BIN, then this checkout's release build,
    then `Y` on PATH."""
    for cand in (explicit, os.environ.get("Y_BIN"), os.path.join(REPO, "target", "release", "Y")):
        if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return shutil.which("Y")


def find_cuda_tool(name):
    """ptxas or nvdisasm: PATH, then the usual CUDA install directories."""
    found = shutil.which(name)
    if found:
        return found
    for base in (os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH"), "/opt/cuda", "/usr/local/cuda"):
        if base:
            cand = os.path.join(base, "bin", name)
            if os.path.isfile(cand) and os.access(cand, os.X_OK):
                return cand
    return None


class ToolError(Exception):
    """A step failed; the message says which and why."""


# ── The compiler ─────────────────────────────────────────────


def compile_ptx(program, y, workdir, cwd=None):
    """PTX with a line table, written into `workdir`."""
    out = os.path.join(workdir, os.path.splitext(os.path.basename(program))[0] + ".ptx")
    run = subprocess.run(
        [y, program, "--emit-ptx", "--lineinfo", "-o", out],
        cwd=cwd or os.getcwd(),
        capture_output=True,
        text=True,
    )
    if run.returncode != 0 or not os.path.isfile(out):
        errors = [l for l in (run.stdout + run.stderr).splitlines() if "[!]" in l or l.startswith("    ")]
        raise ToolError("the Y compiler refused the program:\n" + "\n".join(errors or (run.stdout + run.stderr).splitlines()[-20:]))
    with open(out) as f:
        return out, f.read()


# ── PTX ──────────────────────────────────────────────────────

_FILE = re.compile(r'^\.file\s+(\d+)\s+"((?:[^"\\]|\\.)*)"')
_ENTRY = re.compile(r'^\s*(?:\.visible\s+)?\.entry\s+([A-Za-z_$][\w$]*)\s*\(')
_LOC = re.compile(r'^\s*\.loc\s+(\d+)\s+(\d+)\s+(\d+)')
_DECL = re.compile(r'^\s*\.(reg|local|shared|param|pragma|maxnreg|maxntid|reqntid|minnctapersm)\b')


def _unescape(s):
    return re.sub(r'\\(.)', r'\1', s)


def target_of(ptx):
    m = re.search(r'^\.target\s+(\S+)', ptx, re.M)
    if not m:
        raise ToolError("the PTX module declares no .target")
    return m.group(1).split(",")[0].strip()


def parse_ptx(ptx):
    """{'files': {index: path}, 'kernels': {name: {'lines': {(file, line):
    [instr]}, 'order': [(file, line)], 'unattributed': [instr]}}}.

    An instruction ends at its `;` and may span lines - an operand group in
    braces (`mma.sync .. {%f0, ..}`, a `wmma.store` whose operands continue on
    the next lines). A scope brace is a line of its own."""
    files = {}
    kernels = {}
    current = None
    depth = 0
    loc = None
    pending = ""
    for raw in ptx.splitlines():
        m = _FILE.match(raw)
        if m and current is None:
            files[int(m.group(1))] = _unescape(m.group(2))
            continue
        t = raw.split("//", 1)[0].strip()
        if current is None:
            m = _ENTRY.match(t)
            if m:
                current = kernels.setdefault(m.group(1), {"lines": {}, "order": [], "unattributed": []})
                loc, depth, pending = None, 0, ""
            continue
        if pending:
            pending += " " + t
        elif t == "{":
            depth += 1
            continue
        elif t == "}":
            depth -= 1
            if depth == 0:
                current = None
            continue
        elif depth == 0:
            continue  # the parameter list, .maxnreg
        else:
            m = _LOC.match(t)
            if m:
                loc = (int(m.group(1)), int(m.group(2)))
                continue
            if not t or (t.endswith(":") and ";" not in t) or _DECL.match(t):
                continue
            pending = t
        if ";" in pending:
            insn, pending = pending, ""
            if loc is None:
                current["unattributed"].append(insn)
                continue
            if loc not in current["lines"]:
                current["lines"][loc] = []
                current["order"].append(loc)
            current["lines"][loc].append(insn)
    return {"files": files, "kernels": kernels}


# ── SASS ─────────────────────────────────────────────────────

_FUNC = re.compile(r'^\.text\.([A-Za-z_$][\w$.]*):\s*$')
_SASS_LINE = re.compile(r'//## File "((?:[^"\\]|\\.)*)", line (\d+)')
_SASS_INSN = re.compile(r'^\s*/\*([0-9a-f]+)\*/\s+(.*?)\s*;?\s*$')
_SASS_LABEL = re.compile(r'^\s*(\.L_x_\d+):\s*$')
_BRA_TARGET = re.compile(r'BRA\s+`\((\.L_x_\d+)\)')


def assemble(ptx_path, arch, workdir, ptxas):
    cubin = os.path.join(workdir, "kernel.cubin")
    run = subprocess.run([ptxas, "-arch=" + arch, "-lineinfo", "-o", cubin, ptx_path], capture_output=True, text=True)
    if run.returncode != 0:
        raise ToolError("ptxas -arch=%s rejected the module:\n%s" % (arch, run.stderr.strip()))
    return cubin


def disassemble(cubin, nvdisasm):
    run = subprocess.run([nvdisasm, "-g", "-c", cubin], capture_output=True, text=True)
    if run.returncode != 0:
        raise ToolError("nvdisasm failed:\n" + run.stderr.strip())
    return run.stdout


def parse_sass(sass):
    """{name: {'lines': {(path, line): [(addr, instr)]}, 'order': [...],
    'unattributed': [...], 'padding': n}}."""
    funcs = {}
    current = None
    where = None
    for raw in sass.splitlines():
        m = _FUNC.match(raw)
        if m:
            current = funcs.setdefault(m.group(1), {"insns": []})
            where = None
            continue
        if current is None:
            continue
        m = _SASS_LINE.search(raw)
        if m:
            where = (_unescape(m.group(1)), int(m.group(2)))
            continue
        m = _SASS_LABEL.match(raw)
        if m:
            current["insns"].append(("label", m.group(1), None))
            continue
        m = _SASS_INSN.match(raw)
        if m:
            current["insns"].append((m.group(1), m.group(2).strip(), where))
    out = {}
    for name, f in funcs.items():
        insns = f["insns"]
        # Function-end padding: after the last EXIT, the self-branch ptxas
        # parks a runaway thread on and the NOPs aligning the function.
        padding = set()
        last_exit = max((i for i, x in enumerate(insns) if x[0] != "label" and x[1].split()[0:1] == ["EXIT"]), default=None)
        if last_exit is not None:
            for i in range(last_exit + 1, len(insns)):
                kind, text, _ = insns[i]
                if kind == "label":
                    continue
                if text == "NOP":
                    padding.add(i)
                    continue
                t = _BRA_TARGET.search(text)
                if text.startswith("BRA") and t and i > 0 and insns[i - 1][0] == "label" and insns[i - 1][1] == t.group(1):
                    padding.add(i)
        lines, order, unattributed = {}, [], []
        for i, (addr, text, w) in enumerate(insns):
            if addr == "label" or i in padding:
                continue
            if w is None:
                unattributed.append((addr, text))
                continue
            if w not in lines:
                lines[w] = []
                order.append(w)
            lines[w].append((addr, text))
        out[name] = {"lines": lines, "order": order, "unattributed": unattributed, "padding": len(padding)}
    return out


# ── The map ──────────────────────────────────────────────────


def source_line(path, line, cache={}):
    if path not in cache:
        try:
            with open(path) as f:
                cache[path] = f.read().splitlines()
        except OSError:
            cache[path] = []
    lines = cache[path]
    return lines[line - 1].strip() if 0 < line <= len(lines) else ""


def build_map(program, y=None, cwd=None, sass=True):
    """Compile `program` and return its line map (see `--json`)."""
    y = find_y(y)
    if not y:
        raise ToolError("no Y compiler: build it (cargo build --release) or pass --y")
    work = tempfile.mkdtemp(prefix="ymap_")
    try:
        ptx_path, ptx = compile_ptx(os.path.abspath(program), y, work, cwd)
        arch = target_of(ptx)
        parsed = parse_ptx(ptx)
        files = parsed["files"]
        note = None
        sass_funcs = None
        if sass:
            ptxas, nvdisasm = find_cuda_tool("ptxas"), find_cuda_tool("nvdisasm")
            if not ptxas or not nvdisasm:
                note = "no SASS: %s not found" % " and ".join(t for t, p in (("ptxas", ptxas), ("nvdisasm", nvdisasm)) if not p)
            else:
                sass_funcs = parse_sass(disassemble(assemble(ptx_path, arch, work, ptxas), nvdisasm))
        kernels = []
        for name, k in parsed["kernels"].items():
            s = sass_funcs.get(name) if sass_funcs is not None else None
            keys = []
            for fi, line in k["order"]:
                keys.append((files.get(fi, "<file %d>" % fi), line))
            if s:
                for key in s["order"]:
                    if key not in keys:
                        keys.append(key)
            keys.sort(key=lambda kl: (kl[0] != files.get(1), kl[0], kl[1]))
            ptx_by = {}
            for (fi, line), insns in k["lines"].items():
                ptx_by[(files.get(fi, "<file %d>" % fi), line)] = insns
            entry = {"name": name, "lines": []}
            for path, line in keys:
                row = {
                    "file": path,
                    "line": line,
                    "source": source_line(path, line),
                    "ptx": ptx_by.get((path, line), []),
                }
                if s is not None:
                    row["sass"] = ["%s  %s" % (a, t) for a, t in s["lines"].get((path, line), [])]
                entry["lines"].append(row)
            entry["ptx_unattributed"] = k["unattributed"]
            if s is not None:
                entry["sass_unattributed"] = ["%s  %s" % (a, t) for a, t in s["unattributed"]]
                entry["sass_padding"] = s["padding"]
            kernels.append(entry)
        return {
            "program": os.path.abspath(program),
            "target": arch,
            "files": {str(i): p for i, p in files.items()},
            "sass": note is None and sass,
            "note": note,
            "kernels": kernels,
        }
    finally:
        shutil.rmtree(work, ignore_errors=True)


def select(m, kernel=None, line=None):
    """Keep one kernel and/or one line. `line` is N (the program's own file)
    or FILE:N."""
    if line is not None:
        if isinstance(line, str) and ":" in line:
            fname, n = line.rsplit(":", 1)
            want = lambda r: os.path.basename(r["file"]) == os.path.basename(fname) and r["line"] == int(n)
        else:
            n = int(line)
            want = lambda r: r["file"] == m["files"].get("1") and r["line"] == n
    else:
        want = lambda r: True
    ks = []
    for k in m["kernels"]:
        if kernel and k["name"] != kernel:
            continue
        rows = [r for r in k["lines"] if want(r)]
        if line is not None and not rows:
            continue
        ks.append(dict(k, lines=rows))
    return dict(m, kernels=ks)


def render(m, show_unattributed=True):
    out = []
    for k in m["kernels"]:
        out.append("kernel %s  (%s)" % (k["name"], m["target"]))
        for r in k["lines"]:
            where = r["line"] if r["file"] == m["files"].get("1") else "%s:%d" % (os.path.basename(r["file"]), r["line"])
            out.append("  %4s  %s" % (where, r["source"]))
            out.append("        PTX  %3d" % len(r["ptx"]))
            for i in r["ptx"]:
                out.append("               %s" % i)
            if "sass" in r:
                if r["sass"]:
                    out.append("        SASS %3d" % len(r["sass"]))
                    for i in r["sass"]:
                        out.append("               %s" % i)
                elif r["ptx"]:
                    out.append("        SASS   0  ptxas emitted nothing for this line: removed, or folded into another")
                else:
                    out.append("        SASS   0")
        if show_unattributed and k.get("ptx_unattributed"):
            out.append("  PTX with no line (%d): %s" % (len(k["ptx_unattributed"]), "; ".join(k["ptx_unattributed"][:4])))
        if show_unattributed and k.get("sass_unattributed"):
            out.append("  SASS with no line (%d)" % len(k["sass_unattributed"]))
        out.append("")
    if m.get("note"):
        out.append("(%s - PTX only)" % m["note"])
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Which PTX and SASS each Y line of a kernel became.")
    ap.add_argument("program", help="a .ysu source with at least one `kernel`")
    ap.add_argument("--line", help="one line: N in the program, or FILE:N for an imported file")
    ap.add_argument("--kernel", help="one kernel")
    ap.add_argument("--no-sass", action="store_true", help="PTX only (no ptxas / nvdisasm)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--y", help="the Y compiler to use")
    a = ap.parse_args(argv)
    try:
        m = build_map(a.program, y=a.y, sass=not a.no_sass)
    except ToolError as e:
        print("ymap: %s" % e, file=sys.stderr)
        return 1
    m = select(m, a.kernel, a.line)
    if a.line is not None and not m["kernels"]:
        print("ymap: no kernel instruction comes from line %s" % a.line, file=sys.stderr)
        return 2
    if a.json:
        json.dump(m, sys.stdout, indent=1)
        print()
    else:
        print(render(m, show_unattributed=a.line is None))
    return 0


if __name__ == "__main__":
    sys.exit(main())
