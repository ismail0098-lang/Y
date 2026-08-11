#!/usr/bin/env python3
"""How much of REAL circomlib Y's circom front end compiles, and how big.

Why this exists: every circuit-size figure this repo published was measured on
`Poseidon(2)` or a chain of it, and quoted as a general result ("1.86x fewer
constraints than circom"). It is not general. circomlib's Poseidon is written as
long chains of `<==` linear assignments, which is exactly what
`substitute_linear_constraints` eliminates; most other circuits are already
tight and have nothing to remove. Measured across 31 circomlib gadgets the
geomean is **0.98x - a tie** - with Poseidon the only real win and three
circuits where Y is worse.

It also reports COVERAGE, which nothing else did: the vendored `circomlib/`
in this repo is 7 files, so the test suite could not have noticed that a third
of real circomlib does not compile.

Usage:
    python3 tools/circomlib_coverage.py [--circomlib DIR]

With no --circomlib it shallow-clones iden3/circomlib into a temp dir. Needs
`circom` on PATH for the comparison; without it the Y column still prints.
Build the release binary first: cargo build --release --features zk
"""

import argparse
import glob
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Y_BIN = os.path.join(REPO, "target", "release", "Y")

# One `main` per gadget. Names are prefixed `probe_` because a file called
# `poseidon.circom` shadows circomlib's own: circom resolves an include relative
# to the including file first, so the probe includes ITSELF and reports
# "multiple main components".
CASES = [
    ("aliascheck",    'include "aliascheck.circom";\ncomponent main = AliasCheck();'),
    ("babyadd",       'include "babyjub.circom";\ncomponent main = BabyAdd();'),
    ("babycheck",     'include "babyjub.circom";\ncomponent main = BabyCheck();'),
    ("babypbk",       'include "babyjub.circom";\ncomponent main = BabyPbk();'),
    ("binsum",        'include "binsum.circom";\ncomponent main = BinSum(32, 3);'),
    ("bits2num",      'include "bitify.circom";\ncomponent main = Bits2Num(64);'),
    ("compconstant",  'include "compconstant.circom";\ncomponent main = CompConstant(12345);'),
    ("eddsa",         'include "eddsa.circom";\ncomponent main = EdDSAVerifier(80);'),
    ("eddsamimc",     'include "eddsamimc.circom";\ncomponent main = EdDSAMiMCVerifier();'),
    ("eddsaposeidon", 'include "eddsaposeidon.circom";\ncomponent main = EdDSAPoseidonVerifier();'),
    ("escalarmul",    'include "escalarmul.circom";\ncomponent main = EscalarMul(8, [5299619240641551281634865583518297030282874472190772894086521144482721001553, 16950150798460657717958625567821834550301663161624707787222815936182638968203]);'),
    ("escalarmulany", 'include "escalarmulany.circom";\ncomponent main = EscalarMulAny(8);'),
    ("escalarmulfix", 'include "escalarmulfix.circom";\ncomponent main = EscalarMulFix(8, [5299619240641551281634865583518297030282874472190772894086521144482721001553, 16950150798460657717958625567821834550301663161624707787222815936182638968203]);'),
    ("gates",         'include "gates.circom";\ncomponent main = MultiAND(8);'),
    ("iszero",        'include "comparators.circom";\ncomponent main = IsZero();'),
    ("lessthan",      'include "comparators.circom";\ncomponent main = LessThan(32);'),
    ("mimc7",         'include "mimc.circom";\ncomponent main = MiMC7(91);'),
    ("mimcsponge",    'include "mimcsponge.circom";\ncomponent main = MiMCSponge(2, 220, 1);'),
    ("montgomery",    'include "montgomery.circom";\ncomponent main = Edwards2Montgomery();'),
    ("multiplexer",   'include "multiplexer.circom";\ncomponent main = Multiplexer(4, 8);'),
    ("mux1",          'include "mux1.circom";\ncomponent main = Mux1();'),
    ("mux4",          'include "mux4.circom";\ncomponent main = Mux4();'),
    ("num2bits",      'include "bitify.circom";\ncomponent main = Num2Bits(64);'),
    ("pedersen",      'include "pedersen.circom";\ncomponent main = Pedersen(8);'),
    ("pointbits",     'include "pointbits.circom";\ncomponent main = Point2Bits_Strict();'),
    ("poseidon",      'include "poseidon.circom";\ncomponent main = Poseidon(2);'),
    ("sha256",        'include "sha256/sha256.circom";\ncomponent main = Sha256(64);'),
    ("sign",          'include "sign.circom";\ncomponent main = Sign();'),
    ("smtprocessor",  'include "smt/smtprocessor.circom";\ncomponent main = SMTProcessor(10);'),
    ("smtverifier",   'include "smt/smtverifier.circom";\ncomponent main = SMTVerifier(10);'),
    ("switcher",      'include "switcher.circom";\ncomponent main = Switcher();'),
]


def r1cs_header(path):
    """(n_constraints, n_wires) from an iden3 .r1cs header section."""
    with open(path, "rb") as f:
        d = f.read()
    if d[:4] != b"r1cs":
        return None
    n_sections = struct.unpack("<I", d[8:12])[0]
    off = 12
    for _ in range(n_sections):
        stype, slen = struct.unpack("<IQ", d[off:off + 12])
        off += 12
        if stype == 1:
            fs = struct.unpack("<I", d[off:off + 4])[0]
            q = off + 4 + fs
            n_wires = struct.unpack("<I", d[q:q + 4])[0]
            n_constraints = struct.unpack("<I", d[q + 24:q + 28])[0]
            return n_constraints, n_wires
        off += slen
    return None


def first_error(blob):
    m = re.search(
        r"(?:error|Error|refus\w*|not supported|unsupported|non-quadratic)[^\n]*", blob
    )
    if m:
        return m.group(0).strip()
    lines = [l for l in blob.strip().split("\n") if l.strip()]
    return lines[-1] if lines else "no output"


def classify(err):
    """Group failures by cause, so the list reads as work items not as 11 bugs."""
    if "Neq" in err or "`Eq`" in err:
        return "!=/== over signals"
    if "array literal" in err:
        return "array literal as template arg"
    if "is an array" in err:
        return "array-valued signal port"
    if "`/` by a signal" in err:
        return "/ by a signal"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circomlib", help="path to circomlib/circuits")
    args = ap.parse_args()

    if not os.path.exists(Y_BIN):
        sys.exit(f"build the compiler first: cargo build --release --features zk\n({Y_BIN} not found)")

    tmp = tempfile.mkdtemp(prefix="y_circomlib_cov_")
    lib = args.circomlib
    if not lib:
        clone = os.path.join(tmp, "circomlib")
        print("cloning iden3/circomlib ...", file=sys.stderr)
        subprocess.run(
            ["git", "clone", "-q", "--depth", "1",
             "https://github.com/iden3/circomlib.git", clone],
            check=True,
        )
        lib = os.path.join(clone, "circuits")
    lib = os.path.abspath(lib)

    src = os.path.join(tmp, "src")
    out = os.path.join(tmp, "out")
    os.makedirs(src)
    os.makedirs(out)

    have_circom = shutil.which("circom") is not None
    if not have_circom:
        print("note: `circom` not on PATH; the comparison column is skipped", file=sys.stderr)

    rows = []
    for name, body in CASES:
        p = os.path.join(src, f"probe_{name}.circom")
        with open(p, "w") as f:
            f.write("pragma circom 2.0.0;\n" + body + "\n")

        cres = {}
        if have_circom:
            # BOTH optimisation levels. `--O1` is circom's DEFAULT (its own
            # --help says so) and `--O2` is "full constraint simplification".
            # Comparing only against the default flatters Y: on Poseidon-shaped
            # circuits --O2 produces a SMALLER circuit than Y does, at roughly
            # 2x the compile time. Quoting one number without the level is how
            # this repo published "1.86x smaller than circom" for two days.
            for lvl in ("--O1", "--O2"):
                c = subprocess.run(["circom", p, "--r1cs", lvl, "-o", out, "-l", lib],
                                   capture_output=True, text=True)
                rp = os.path.join(out, f"probe_{name}.r1cs")
                if c.returncode == 0 and os.path.exists(rp):
                    cres[lvl] = r1cs_header(rp)
                    os.remove(rp)

        y = subprocess.run([Y_BIN, p, "--target=r1cs", "-l", lib],
                           capture_output=True, text=True)
        yp = p.replace(".circom", ".r1cs")
        yres = r1cs_header(yp) if y.returncode == 0 and os.path.exists(yp) else None
        err = "" if yres else first_error(y.stdout + y.stderr)
        rows.append((name, cres, yres, err))

    print(f"\n{'circuit':<16}{'circom --O1':>14}{'circom --O2':>14}{'Y':>12}{'vs O1':>8}{'vs O2':>8}  note")
    print("-" * 100)
    for name, c, yv, err in rows:
        o1 = c.get("--O1")
        o2 = c.get("--O2")
        f1 = f"{o1[0]:,}" if o1 else "-"
        f2 = f"{o2[0]:,}" if o2 else "-"
        ys = f"{yv[0]:,}" if yv else "REFUSED"
        r1 = f"{o1[0] / yv[0]:.2f}x" if (o1 and yv) else ""
        r2 = f"{o2[0] / yv[0]:.2f}x" if (o2 and yv) else ""
        note = "" if yv else f"[{classify(err)}]"
        print(f"{name:<16}{f1:>14}{f2:>14}{ys:>12}{r1:>8}{r2:>8}  {note}")
    print("-" * 100)

    n_y = sum(1 for r in rows if r[2])
    n_c = sum(1 for r in rows if r[1].get("--O1"))
    print(f"coverage: Y {n_y}/{len(rows)}" + (f", circom {n_c}/{len(rows)}" if have_circom else ""))

    for lvl in ("--O1", "--O2"):
        # A circuit can reduce to ZERO constraints (circom --O2 does that to
        # `Bits2Num`, which is entirely linear). Those carry no ratio - a
        # geomean cannot take log(0) - so they are counted separately rather
        # than dropped silently.
        both = [(n, c[lvl][0], y[0]) for n, c, y, _ in rows
                if c.get(lvl) and y and c[lvl][0] > 0 and y[0] > 0]
        zeroed = [n for n, c, y, _ in rows if c.get(lvl) and y and c[lvl][0] == 0]
        if not both:
            continue
        ratios = [c / y for _, c, y in both]
        gm = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
        wins = sum(1 for r in ratios if r > 1.05)
        ties = sum(1 for r in ratios if 0.95 <= r <= 1.05)
        loss = sum(1 for r in ratios if r < 0.95)
        tag = "(circom default)" if lvl == "--O1" else "(circom best)   "
        extra = f"  [+{len(zeroed)} reduced to 0 by circom: {', '.join(zeroed)}]" if zeroed else ""
        print(f"size vs {lvl} {tag}: geomean {gm:.3f}x over {len(both)}"
              f"  (win {wins} / tie {ties} / loss {loss})"
              f"   totals circom {sum(c for _, c, _ in both):,} / Y {sum(y for _, _, y in both):,}{extra}")

    failed = [(n, classify(e)) for n, _, y, e in rows if not y]
    if failed:
        causes = {}
        for n, cause in failed:
            causes.setdefault(cause, []).append(n)
        print("\nrefused, by cause:")
        for cause, names in sorted(causes.items(), key=lambda kv: -len(kv[1])):
            print(f"  {len(names):>2}  {cause:<32} {', '.join(names)}")

    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
