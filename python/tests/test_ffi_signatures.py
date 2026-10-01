"""The Python bindings' `argtypes` must match the Rust `extern "C"` signatures.

`y_autotune_search_space_json` is declared in Rust as

    pub unsafe extern "C" fn y_autotune_search_space_json(
        m: u32, n: u32, k: u32, is_fp8: bool) -> *mut c_char

and the bindings declared THREE parameters for it. ctypes fills only the
registers it is told about, so the callee read whatever happened to be in the
fourth argument register - observed flipping between True and False across
calls within a single process, i.e. the precision of the search space was
undefined. Nothing raises: the call returns, the JSON parses, and one field of
it is a coin flip.

An arity mismatch on an ABI boundary cannot be caught by running the code, so
it is checked against the Rust source instead. This is the same shape as
`tests/ptx_portability.rs::no_source_file_hardcodes_a_target_above_the_floor`:
a source-level guard for a class of bug whose symptom is silence.
"""

import ctypes
import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
# `python/` lives INSIDE the cargo project since the merge 37651fb, so the Rust
# source is `<repo>/src/c_api.rs`. This said `<repo>/Y/src/c_api.rs` - the
# layout of the old outer checkout, where `python/` was a sibling of the cargo
# project `Y/` - and `test_c_api_source_is_readable` below is what reported it.
C_API = REPO / "src" / "c_api.rs"

SIGNATURE = re.compile(
    r'pub\s+(?:unsafe\s+)?extern\s+"C"\s+fn\s+(y_\w+)\s*\(', re.MULTILINE
)


def rust_signatures():
    """{symbol: parameter count} for every exported C function."""
    text = C_API.read_text()
    out = {}
    for m in SIGNATURE.finditer(text):
        name = m.group(1)
        # Walk to the matching close paren; parameters may span lines.
        i = m.end()
        depth = 1
        start = i
        while depth:
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
            i += 1
        params = text[start : i - 1]
        # Split on top-level commas only (a param type can contain none here,
        # but *mut *mut c_char and generics would - be conservative).
        parts, depth, buf = [], 0, ""
        for ch in params:
            if ch in "(<[":
                depth += 1
            elif ch in ")>]":
                depth -= 1
            if ch == "," and depth == 0:
                parts.append(buf)
                buf = ""
            else:
                buf += ch
        if buf.strip():
            parts.append(buf)
        out[name] = len([p for p in parts if p.strip()])
    return out


class TestFfiSignatures(unittest.TestCase):
    def test_c_api_source_is_readable(self):
        """The control: if the source moved, every assertion below is vacuous."""
        self.assertTrue(C_API.exists(), f"{C_API} not found")
        sigs = rust_signatures()
        self.assertIn("y_autotune_search_space_json", sigs)
        self.assertGreaterEqual(len(sigs), 5, f"only found {sigs}")

    def test_every_declared_argtypes_matches_rust(self):
        from y_lang.compiler import YCompilerLib

        sigs = rust_signatures()
        lib = YCompilerLib.get_instance()

        checked = 0
        for name, arity in sigs.items():
            fn = getattr(lib.lib, name, None)
            if fn is None:
                continue
            argtypes = getattr(fn, "argtypes", None)
            if argtypes is None:
                # Undeclared is a separate (looser) problem - ctypes then
                # guesses per call site. Only declared signatures are pinned
                # here, so that this test says something exact.
                continue
            checked += 1
            self.assertEqual(
                len(argtypes),
                arity,
                f"{name}: bindings declare {len(argtypes)} parameters, "
                f"Rust takes {arity}. ctypes fills only what it is told "
                f"about, so the extra parameter is read from an "
                f"uninitialised register.",
            )
        self.assertGreater(checked, 0, "no declared signatures were compared")

    def test_bool_parameters_are_declared_as_bool(self):
        """`is_fp8: bool` must not be declared as a c_uint32 or omitted."""
        from y_lang.compiler import YCompilerLib

        lib = YCompilerLib.get_instance()
        for name in ("y_autotune_search_space_json", "y_autotune_select_config_json"):
            fn = getattr(lib.lib, name)
            self.assertEqual(
                fn.argtypes[-1],
                ctypes.c_bool,
                f"{name}'s trailing `is_fp8: bool` must be declared c_bool",
            )

    def test_the_loaded_library_is_not_older_than_the_c_api(self):
        """A stale `liby.so` answers instead of raising, so staleness is checked.

        `_find_liby` used to look only in `<package>/../target`, which is NOT
        where `cargo build --release` writes - the cargo project is `Y/`. A
        `liby.so` from an abandoned build layout was sitting there, so the
        whole package ran against a compiler 25 days older than `src/` with no
        symptom. Missing symbols raise; stale ones just answer.
        """
        from y_lang.compiler import YCompilerLib

        lib_path = Path(YCompilerLib.get_instance().lib_path)
        self.assertTrue(lib_path.exists(), lib_path)
        self.assertGreaterEqual(
            lib_path.stat().st_mtime,
            C_API.stat().st_mtime,
            f"{lib_path} is older than {C_API}: the bindings are running "
            f"against a stale build. Run `cargo build --release` in Y/.",
        )


if __name__ == "__main__":
    unittest.main()
