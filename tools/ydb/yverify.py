#!/usr/bin/env python3
"""What covers a line of a Y program: what the compiler proved, checked or
assumed there, and what - if anything - checks the code the line became.

    yverify.py prog.ysu 14            the report for line 14
    yverify.py prog.ysu 14 --y PATH   with another Y compiler

`ydb`'s `verify` command is this report, given the facts the program being
debugged carries. Run on its own it compiles the source with
`Y --emit-guarantees` instead.

Two sources, kept apart because they are different kinds of evidence:

  * The COMPILER's facts (`src/guarantees.rs`): what the type checker proved
    (an index in bounds, a loop invariant, by z3), what it checked, what it
    took on trust (a `@bounds` it could not check), and what the backend
    substituted. Each proof names the assumptions it used.
  * The REPOSITORY's evidence about the code a kernel became: the Rocq proofs
    of a lowering this kernel was given (`LOWERINGS`, below), and the standing
    results of `tools/ptxas_tval` - per-translation validation that ptxas's
    SASS stores what the PTX stores. Those are about particular PTX, so they
    are credited to a kernel only when this compile's kernel is the same PTX,
    instruction for instruction, as the committed artifact they are about.
    Matching by kernel name alone would credit a different kernel with
    another's proof.

It reports and decides nothing: every line is read from the compiler, the
proofs, `regress.sh` or a compile.
"""
import argparse
import fnmatch
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

import ymap  # noqa: E402  (the same compiler discovery)


class ToolError(Exception):
    pass


# ── the compiler's facts ─────────────────────────────────────

def facts_from_compiler(program, y=None, cwd=None):
    """The guarantee table `Y --emit-guarantees` writes for `program`."""
    y = ymap.find_y(y)
    if not y:
        raise ToolError("no Y compiler: build it (cargo build --release) or pass --y")
    work = tempfile.mkdtemp(prefix="yverify_")
    try:
        out = os.path.join(work, "g.json")
        run = subprocess.run(
            [y, os.path.abspath(program), "--emit-guarantees", "-o", out],
            cwd=cwd or os.getcwd(), capture_output=True, text=True,
        )
        if run.returncode != 0 or not os.path.isfile(out):
            errs = [l for l in (run.stdout + run.stderr).splitlines() if "[Error]" in l or "[!]" in l]
            raise ToolError("the Y compiler refused the program, so it established nothing:\n"
                            + "\n".join(errs or (run.stdout + run.stderr).splitlines()[-10:]))
        with open(out) as f:
            return json.load(f)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _same_file(a, b):
    if not a or not b:
        return False
    try:
        return os.path.realpath(a) == os.path.realpath(b)
    except OSError:
        return os.path.basename(a) == os.path.basename(b)


def enclosing_item(g, file, line):
    """The innermost function or kernel spanning `file:line`, or None."""
    best = None
    for it in g.get("items", []):
        if _same_file(it["file"], file) and it["line"] <= line <= it["end"]:
            if best is None or (it["end"] - it["line"]) < (best["end"] - best["line"]):
                best = it
    return best


def facts_at(g, file, line):
    """The facts about `file:line`: those ON it first (by column), then those
    spanning it, innermost first."""
    on, spanning = [], []
    for f in g.get("facts", []):
        if not _same_file(f["file"], file):
            continue
        if f["line"] == line and f["end"] == line:
            on.append(f)
        elif f["line"] <= line <= f["end"]:
            spanning.append(f)
    on.sort(key=lambda f: (f["col"], f["kind"]))
    spanning.sort(key=lambda f: (f["end"] - f["line"], f["line"]))
    return on + spanning


# ── the repository's evidence about a kernel's code ──────────

# A lowering the PTX backend gives a whole kernel, recognised by the marker it
# writes into the kernel, and the proofs about that lowering. Every theorem is
# named so `tests/ydb_verify.rs` can check it exists, and every marker is
# checked against `src/ptx_emitter.rs`; a lowering with no proof says so.
LOWERINGS = [
    {
        "marker": "[Y INT8 TENSOR CORE GEMM]",
        "name": "Y's int8 tensor-core GEMM",
        "proofs": [
            ("proofs/Int8GemmSchedule.v", ["the_guard_is_what_confines_a_warp_to_its_own_tile",
                                           "the_atomic_reduction_is_order_independent"],
             "the schedule: each warp writes only its own tile, and the atomic reduction gives the "
             "same int32 result in any order, for every block size, grid and split factor"),
            ("proofs/Int8GemmExact.v", ["the_emitted_int8_gemm_holds_the_source_dot_products",
                                        "the_split_k_accumulation_is_exact_in_int32"],
             "the value: C holds the exact int32 dot products - for K up to 133,120, which the "
             "compiler checks, and into a C the host has zeroed, which nothing checks"),
        ],
        "also": [("[Y INT8 GEMM] the fused epilogue STORES",
                  "the fused epilogue dequantises to f32 after the exact accumulation; its f32 "
                  "multiply and add round, and no proof covers them")],
    },
    {
        "marker": "[Y TENSOR CORE GEMM]",
        "name": "Y's f16 tensor-core GEMM",
        "proofs": [
            ("proofs/GpuWarpTiling.v", ["cta_rows_written_exactly_once"],
             "the schedule only: every row of each CTA tile is written by exactly one warp, for "
             "every tile the compiler accepts. The f16 products are accumulated in f32, which "
             "rounds, so nothing proves the values"),
        ],
    },
    {
        "marker": "[Y FP8 TENSOR CORE GEMM]",
        "name": "Y's FP8 tensor-core GEMM",
        "proofs": [
            ("proofs/GpuWarpTiling.v", ["cta_rows_written_exactly_once"],
             "the schedule only: every row of each CTA tile is written by exactly one warp. The "
             "values round, so nothing proves them"),
        ],
    },
    {
        "marker": "[Y FUSED LINEAR+SWIGLU GEMM]",
        "name": "Y's fused SwiGLU GEMM",
        "proofs": [
            ("proofs/GpuWarpTiling.v", ["cta_rows_written_exactly_once"],
             "the schedule only: every row of each CTA tile is written by exactly one warp. The "
             "values round, so nothing proves them"),
        ],
    },
    {"marker": "[Y PAGED DECODE ATTENTION]", "name": "Y's paged decode attention", "proofs": []},
    {"marker": "[Y FUSED ROPE]", "name": "Y's fused RoPE kernel", "proofs": []},
]


# A proof about one committed kernel, rather than about a lowering: the
# fixture whose PTX it is about, and its capstone theorem. A proof is credited
# to a kernel only through this table - a proof may name a fixture for other
# reasons (`GpuWarpTiling.v` names `tests/gemm_f16_1024.ysu` as the instance of
# the bug it refutes). `tests/ydb_verify.rs` checks each entry: the proof names
# the fixture, and the theorem exists.
FIXTURE_PROOFS = {
    "proofs/ExactPvExact.v": ("exact_pv", "the_emitted_exact_pv_holds_the_source_dot_product",
                              "conditions on the arguments the kernel is launched with: the "
                              "compiler cannot check them, and nothing checks them at launch"),
}


def theorem_hypotheses(path, theorem):
    """The comments inside `theorem`'s statement - in this repository, each
    hypothesis is labelled in the theorem's own words."""
    try:
        with open(path) as f:
            text = f.read()
    except OSError:
        return []
    m = re.search(r"Theorem %s\s*:(.*?)\bProof\." % re.escape(theorem), text, re.S)
    if not m:
        return []
    return [" ".join(c.split()) for c in re.findall(r"\(\*(.*?)\*\)", m.group(1), re.S)]


def proof_title(path):
    """A proof file's title: its `(** * ... *)` heading."""
    try:
        with open(path) as f:
            for l in f:
                m = re.match(r"\(\*\* \* (.*?)\s*(\*\))?\s*$", l)
                if m:
                    return m.group(1).strip()
    except OSError:
        pass
    return None


def ptx_parts(ptx):
    """(header, module, entries) of a PTX module, with comments, `.loc`,
    `.file` and blank lines dropped: the instructions, which is what the
    comparison is about. `header` is `.version`/`.target`/`.address_size`;
    `module` the other module-level lines; `entries` name -> lines."""
    header, module, entries = [], [], {}
    current, depth = None, 0
    for raw in ptx.splitlines():
        line = raw.split("//", 1)[0].strip()
        if not line or line.startswith(".loc") or line.startswith(".file"):
            continue
        m = re.match(r"(?:\.visible\s+)?\.entry\s+([A-Za-z_$][\w$]*)", line)
        if current is None and m:
            current, depth = m.group(1), 0
            entries[current] = []
        if current is not None:
            entries[current].append(line)
            depth += line.count("{") - line.count("}")
            if depth == 0 and "}" in line:
                current = None
            continue
        if re.match(r"\.(version|target|address_size)\b", line):
            header.append(line)
        else:
            module.append(line)
    return header, module, entries


def same_kernel(this_ptx, artifact_ptx, name):
    """(same?, why not): the kernel `name` in this compile is the artifact's,
    instruction for instruction, at the same target, with every module-level
    declaration the artifact has."""
    th, tm, te = ptx_parts(this_ptx)
    ah, am, ae = ptx_parts(artifact_ptx)
    if name not in ae:
        return False, "the artifact has no kernel `%s`" % name
    if name not in te:
        return False, "this compile has no kernel `%s`" % name
    if th != ah:
        return False, "this compile declares %s, the artifact %s" % (" ".join(th), " ".join(ah))
    if te[name] != ae[name]:
        return False, "the kernel's instructions differ from the artifact's"
    missing = [l for l in am if l not in tm]
    if missing:
        return False, "the artifact declares %s, which this compile does not" % missing[0]
    return True, ""


def standing_rows(text):
    """Every row `tools/ptxas_tval/regress.sh` asserts, as (path, validator,
    verdict): `("o1/exact_pv", "loopval", "VALIDATED")`. Read from the script's
    own `for` lists and `case` arms, so a row is credited only as the script
    asserts it."""
    rows = []
    text = text.replace("\\\n", " ")
    for m in re.finditer(r"^for (\w+) in (.*?); do\n(.*?)^done", text, re.S | re.M):
        var, items_txt, body = m.groups()
        items = [a or b for a, b in re.findall(r'"([^"]*)"|(\S+)', items_txt)]
        case = re.search(r'case "\$(\w+)" in(.*?)esac', body, re.S)
        if not case:
            continue
        subj_var, arms_txt = case.groups()
        arms = []
        for am in re.finditer(r"^\s*([^)\n]+)\)\s*(.*?);;", arms_txt, re.M):
            g = re.search(r"grep -q '([^']*)'", am.group(2))
            arms.append(([p.strip() for p in am.group(1).split("|")], g.group(1) if g else None))
        literal = re.search(r"python3 (\w+)\.py", body)
        for item in items:
            words = item.split()
            paths = [w for w in words if re.match(r"^\w+/\w+(\.\w+)?$", w)]
            if not paths:
                continue
            path = re.sub(r"\.\w+$", "", paths[0])
            if literal:
                validator = literal.group(1)
            elif re.search(r'python3 "\$1\.py"', body):
                validator = words[0]
            else:
                continue
            if subj_var == var:
                subject = item
            else:
                a = re.search(r'%s=\$\(basename "\$(\d)"(?: ([.\w]+))?\)' % subj_var, body)
                if not a:
                    continue
                k = int(a.group(1))
                subject = os.path.basename(words[k - 1] if len(words) >= k else words[0])
                if a.group(2) and subject.endswith(a.group(2)):
                    subject = subject[: -len(a.group(2))]
            for pats, rx in arms:
                if any(fnmatch.fnmatchcase(subject, p) for p in pats):
                    mm = re.match(r"\^(VALIDATED|UNPROVED|REFUSED)", rx or "")
                    rows.append((path, validator, mm.group(1) if mm else "?"))
                    break
    return rows


# What each corpus directory holds: which ptxas made the SASS under validation.
TVAL_DIRS = {
    "corpus": "the SASS ptxas makes at its default level (-O3) for %s",
    "o1": "the SASS ptxas -O1 makes for %s",
    "smut": "the SASS for %s committed in tools/ptxas_tval/smut/ (whether today's ptxas still "
            "makes it is not checked)",
}


def artifacts(repo=REPO):
    """Committed PTX that repository evidence is about: stem -> {path, rows,
    proofs}. A tval row's PTX is `tests/<stem>.ptx` for corpus/ and o1/
    (`build_corpus.sh` copies it) and the committed file for smut/; a proof is
    about `tests/<stem>` when it names that fixture."""
    out = {}
    regress = os.path.join(repo, "tools", "ptxas_tval", "regress.sh")
    if os.path.isfile(regress):
        with open(regress) as f:
            for path, validator, verdict in standing_rows(f.read()):
                d, stem = path.split("/", 1)
                if d not in TVAL_DIRS:
                    continue
                ptx = (os.path.join(repo, "tools", "ptxas_tval", d, stem + ".ptx") if d == "smut"
                       else os.path.join(repo, "tests", stem + ".ptx"))
                if not os.path.isfile(ptx):
                    continue  # a derived twin, not a committed kernel
                a = out.setdefault((ptx, stem), {"ptx": ptx, "stem": stem, "rows": [], "proofs": []})
                a["rows"].append((path, validator, verdict))
    for proof, (stem, _, _) in sorted(FIXTURE_PROOFS.items()):
        ptx = os.path.join(repo, "tests", stem + ".ptx")
        if os.path.isfile(ptx) and os.path.isfile(os.path.join(repo, proof)):
            a = out.setdefault((ptx, stem), {"ptx": ptx, "stem": stem, "rows": [], "proofs": []})
            a["proofs"].append(proof)
    return list(out.values())


def compile_ptx(program, y=None, cwd=None):
    """The PTX `Y --emit-ptx` makes of `program` here: (text, None) or
    (None, why it did not)."""
    y = ymap.find_y(y)
    if not y:
        return None, "no Y compiler"
    work = tempfile.mkdtemp(prefix="yverify_")
    try:
        out = os.path.join(work, "p.ptx")
        run = subprocess.run([y, os.path.abspath(program), "--emit-ptx", "-o", out],
                             cwd=cwd or os.getcwd(), capture_output=True, text=True)
        if run.returncode != 0 or not os.path.isfile(out):
            errs = [l.strip() for l in (run.stdout + run.stderr).splitlines() if "[!]" in l or "[Error]" in l]
            return None, (errs[0] if errs else "Y --emit-ptx refused it")
        with open(out) as f:
            return f.read(), None
    finally:
        shutil.rmtree(work, ignore_errors=True)


def gpu_evidence(kernel, ptx, repo=REPO):
    """Lines about the code the GPU runs for `kernel`, from this compile's
    PTX `ptx` and the repository's evidence."""
    lines = []
    header, _, entries = ptx_parts(ptx)
    target = next((h.split()[1] for h in header if h.startswith(".target")), "?")
    if kernel not in entries:
        return ["kernel %s does not appear in the PTX" % kernel], False
    # The entry's own comments carry the lowering markers.
    body = _entry_raw(ptx, kernel)
    validated = False
    for low in LOWERINGS:
        if low["marker"] not in body:
            continue
        lines.append(("REPLACED", "kernel %s is replaced wholesale by %s: its body's lines become no "
                      "PTX of their own" % (kernel, low["name"])))
        if not low["proofs"]:
            lines.append(("NOT PROVED", "no proof covers %s" % low["name"]))
        for path, theorems, claim in low["proofs"]:
            lines.append(("PROVED", "%s: %s (%s)" % (path, claim, ", ".join(theorems))))
        for marker, note in low.get("also", []):
            if marker in body:
                lines.append(("NOT PROVED", note))
    for a in artifacts(repo):
        try:
            with open(a["ptx"]) as f:
                art = f.read()
        except OSError:
            continue
        same, why = same_kernel(ptx, art, kernel)
        rel = os.path.relpath(a["ptx"], repo)
        if not same:
            if kernel in ptx_parts(art)[2]:
                lines.append(("NOT COVERED", "%s's proofs and validation are about the kernel %s holds; "
                              "this one differs: %s" % (rel, rel, why)))
            continue
        lines.append(("SAME PTX", "this compile's kernel is %s's, instruction for instruction" % rel))
        for p in a["proofs"]:
            title = proof_title(os.path.join(repo, p))
            _, theorem, caveat = FIXTURE_PROOFS[p]
            hyps = theorem_hypotheses(os.path.join(repo, p), theorem)
            text = "%s%s (%s)" % (p, ": " + title if title else "", theorem)
            if hyps:
                text += ". It assumes, in its own words: %s - %s" % ("; ".join(hyps), caveat)
            lines.append(("PROVED", text))
        for path, validator, verdict in a["rows"]:
            d = path.split("/", 1)[0]
            arch = next((h.split()[1] for h in ptx_parts(art)[0] if h.startswith(".target")), "?")
            where = TVAL_DIRS[d] % arch
            if verdict == "VALIDATED":
                validated = True
                lines.append(("VALIDATED", "%s: %s.py proves it stores what this PTX stores - a "
                              "standing result of tools/ptxas_tval/regress.sh (row %s), not re-run here"
                              % (where, validator, path)))
            else:
                lines.append((verdict, "%s: %s.py does not validate it (regress.sh row %s)"
                              % (where, validator, path)))
    lines.append(("TRUSTED", "ptxas: %s" % (
        "outside the validated level and architecture above, and the CUDA driver's JIT if it "
        "compiles this PTX itself" if validated else
        "the SASS is not checked against the PTX for this kernel")))
    lines.append(("TRUSTED", "the GPU executing its instruction set"))
    return [("", "target %s" % target)] + lines, validated


def _entry_raw(ptx, name):
    """The kernel's lines as written, comments included."""
    out, inside, depth = [], False, 0
    for raw in ptx.splitlines():
        code = raw.split("//", 1)[0]
        if not inside and re.match(r"\s*(?:\.visible\s+)?\.entry\s+%s\b" % re.escape(name), code):
            inside, depth = True, 0
        if inside:
            out.append(raw)
            depth += code.count("{") - code.count("}")
            if depth == 0 and "}" in code:
                break
    return "\n".join(out)


# ── the report ───────────────────────────────────────────────

STATUS = {
    "proved": "PROVED", "checked": "CHECKED", "run-time": "RUN-TIME", "tested": "TESTED",
    "trusted": "TRUSTED", "not-checked": "NOT CHECKED", "unverified": "UNVERIFIED",
}


def _wrap(tag, text, width=100):
    head = "  %-13s" % tag
    return textwrap.fill(text, width=width, initial_indent=head, subsequent_indent=" " * len(head))


def _where(f, line):
    if f["line"] == line and f["end"] == line:
        return ""
    if f["kind"] == "invariant":
        return " (the loop at line %d)" % f["line"]
    return " (lines %d-%d)" % (f["line"], f["end"])


def report(program, line, g, facts_source, file=None, host=None, gpu_ptx=None, gpu_error=None, repo=REPO):
    """The report for `file:line` (the program's own file by default), as text.

    `g` is the guarantee table, `facts_source` says where it came from.
    `host` describes the binary being debugged (`clang -O0`), or is None when
    there is none. `gpu_ptx` is this program's PTX, or None with `gpu_error`
    saying why there is none."""
    file = file or program
    out = []
    src = ymap.source_line(file, line) if os.path.isfile(file) else ""
    out.append("%s:%d  %s" % (os.path.basename(file), line, src))
    item = enclosing_item(g, file, line)
    if item is None:
        out.append("  not inside a function or kernel: no code comes from this line")
        return "\n".join(out)
    out.append("in %s %s (%s:%d-%d)" % (item["kind"], item["name"], os.path.basename(file), item["line"], item["end"]))
    facts = facts_at(g, file, line)
    out.append("")
    out.append("What the compiler established (%s):" % facts_source)
    groups = []
    for f in facts:
        key = (f["kind"], f["status"], f["what"], f["detail"], f["line"], f["end"])
        if groups and groups[-1][0] == key:
            groups[-1][1] += 1
            continue
        groups.append([key, 1, f])
    for _, n, f in groups:
        times = " (%d accesses on this line)" % n if n > 1 else ""
        out.append(_wrap(STATUS.get(f["status"], f["status"].upper()),
                         "%s%s%s: %s" % (f["what"], times, _where(f, line), f["detail"])))
        for a in f["rests_on"]:
            at = " (%s:%d)" % (os.path.basename(a["file"]), a["line"]) if a["line"] else ""
            out.append(_wrap("", "assumes, without checking: %s%s" % (a["what"], at)))
    if host is not None:
        out.append("")
        out.append("The code this process runs (%s, from the LLVM IR):" % host)
        out.append(_wrap("TRUSTED", "everything below the LLVM IR - clang, the assembler, the linker "
                         "and the processor. Nothing checks the machine code against the IR."))
    if item["kind"] == "kernel":
        out.append("")
        if gpu_ptx is None:
            out.append("The code the GPU runs:")
            out.append(_wrap("", "unknown: %s" % (gpu_error or "the program was not compiled for the GPU")))
        else:
            ev, _ = gpu_evidence(item["name"], gpu_ptx, repo)
            target = ev[0][1].split()[-1]
            out.append("The code the GPU runs (Y --emit-ptx for %s, then ptxas):" % target)
            for tag, text in ev[1:]:
                out.append(_wrap(tag, text))
    return "\n".join(out)


def selftest(repo=REPO):
    """Check the tables above against the repository they describe; each
    check carries a control that perturbs its input and must be reported.
    Returns the failures."""
    bad = []
    emitter = os.path.join(repo, "src", "ptx_emitter.rs")
    with open(emitter) as f:
        emitter_src = f.read()

    def theorem_exists(path, name):
        try:
            with open(os.path.join(repo, path)) as f:
                return re.search(r"^\s*(Theorem|Lemma)\s+%s\b" % re.escape(name), f.read(), re.M) is not None
        except OSError:
            return False

    for low in LOWERINGS:
        if low["marker"] not in emitter_src:
            bad.append("the marker %s is not written by src/ptx_emitter.rs" % low["marker"])
        for marker, _ in low.get("also", []):
            if marker not in emitter_src:
                bad.append("the marker %s is not written by src/ptx_emitter.rs" % marker)
        for path, theorems, _ in low["proofs"]:
            for t in theorems:
                if not theorem_exists(path, t):
                    bad.append("%s has no theorem %s" % (path, t))
    for path, (stem, theorem, _) in FIXTURE_PROOFS.items():
        if not theorem_exists(path, theorem):
            bad.append("%s has no theorem %s" % (path, theorem))
        with open(os.path.join(repo, path)) as f:
            if "tests/%s." % stem not in f.read():
                bad.append("%s does not name tests/%s, the fixture it is credited for" % (path, stem))
        if not theorem_hypotheses(os.path.join(repo, path), theorem):
            bad.append("%s's %s states no hypothesis comment to quote" % (path, theorem))
    # Controls: a theorem and a marker that do not exist must be reported.
    if theorem_exists("proofs/ExactPvExact.v", "no_such_theorem_anywhere"):
        bad.append("CONTROL: theorem_exists found a theorem that does not exist")
    if "[Y NO SUCH LOWERING]" in emitter_src:
        bad.append("CONTROL: the marker check would accept anything")

    # The standing results this reads, against the rows regress.sh asserts.
    with open(os.path.join(repo, "tools", "ptxas_tval", "regress.sh")) as f:
        rows = standing_rows(f.read())
    for want in [("o1/exact_pv", "loopval", "VALIDATED"), ("o1/exact_pv", "nestval", "VALIDATED"),
                 ("o1/naive_gemm_f32", "loopval", "VALIDATED"),
                 ("o1/naive_gemm_f32_muladd", "loopval", "UNPROVED"),
                 ("o1/y_cpu_matmul", "nestval", "VALIDATED"),
                 ("corpus/ptx_integer_ops", "tval", "VALIDATED"),
                 ("smut/smem_roundtrip", "smemval", "VALIDATED"),
                 ("fma/plain", "tval", "UNPROVED")]:
        if want not in rows:
            bad.append("regress.sh's row %s is not read as %s %s" % want)
    # Control: a row the script asserts the other way must read the other way.
    flipped = standing_rows("for t in \"loopval o1/k\"; do\n  set -- $t\n  name=$(basename \"$2\")\n"
                            "  out=$(python3 \"$1.py\" x)\n  case \"$name\" in\n"
                            "    k) echo \"$out\" | grep -q '^UNPROVED' || bad=1 ;;\n"
                            "    *) echo \"$out\" | grep -q '^VALIDATED' || bad=1 ;;\n  esac\ndone\n")
    if flipped != [("o1/k", "loopval", "UNPROVED")]:
        bad.append("CONTROL: a row asserted UNPROVED was read as %s" % flipped)
    committed = [a for a in artifacts(repo) if a["rows"]]
    if len(committed) < 8:
        bad.append("only %d committed kernels have standing results (want at least 8)" % len(committed))

    # The comparison: a kernel is the same as itself, and not after one
    # instruction changes.
    art = os.path.join(repo, "tests", "exact_pv.ptx")
    with open(art) as f:
        ptx = f.read()
    if not same_kernel(ptx, ptx, "exact_pv")[0]:
        bad.append("same_kernel says tests/exact_pv.ptx differs from itself")
    if not same_kernel(ptx.replace("\n", "\n// a comment\n", 3), ptx, "exact_pv")[0]:
        bad.append("same_kernel is not blind to comments")
    retargeted = re.sub(r"^\.target\s+sm_\d+", ".target sm_80", ptx, count=1, flags=re.M)
    if retargeted == ptx or same_kernel(retargeted, ptx, "exact_pv")[0]:
        bad.append("CONTROL: same_kernel accepted a kernel compiled for another target")
    altered = re.sub(r"(add\.s64\s+%rd\d+, )(%rd\d+), (%rd\d+)", r"\1\3, \2", ptx, count=1)
    if altered == ptx:
        bad.append("CONTROL: the exact_pv instruction to alter was not found")
    elif same_kernel(altered, ptx, "exact_pv")[0]:
        bad.append("CONTROL: same_kernel accepted a kernel with an instruction changed")
    return bad


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if argv == ["--selftest"]:
        bad = selftest()
        for b in bad:
            print("FAIL: " + b)
        print("selftest: %s" % ("ok" if not bad else "%d failure(s)" % len(bad)))
        return 1 if bad else 0
    ap = argparse.ArgumentParser(prog="yverify", description=__doc__.split("\n\n")[0])
    ap.add_argument("program")
    ap.add_argument("line", type=int)
    ap.add_argument("--y", help="the Y compiler to use")
    a = ap.parse_args(argv)
    try:
        g = facts_from_compiler(a.program, a.y)
    except ToolError as e:
        print("yverify: %s" % e, file=sys.stderr)
        return 1
    item = enclosing_item(g, a.program, a.line)
    ptx, err = (None, None)
    if item and item["kind"] == "kernel":
        ptx, err = compile_ptx(a.program, a.y)
    print(report(a.program, a.line, g, "Y --emit-guarantees", gpu_ptx=ptx, gpu_error=err))
    return 0


if __name__ == "__main__":
    sys.exit(main())
