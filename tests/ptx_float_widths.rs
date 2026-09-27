//! `F64` and `F16` in a PTX kernel are what they say, or they are refused.
//!
//! `ScalarTy::from_name` had no `F64` or `F16` arm and returned `None`, and
//! every caller read `None` as "no annotation". So
//!
//! ```text
//! kernel k(Out: GlobalMemory<F64>) {
//!     let x: F64 = 3.0;
//!     store(Out, x + x);
//! }
//! ```
//!
//! emitted `mov.f32` / `add.f32` / `st.global.f32` - four bytes into an
//! eight-byte slot - and the card read back `00 00 c0 40 ab ab ab ab`, i.e.
//! -2.53e-98. The `F16` version wrote four bytes into a two-byte slot: element
//! 0 read back +0.0 and element 1, which the program never wrote, became
//! `0x40c0`. Both exited 0 and `ptxas` accepted both. The same `None` reached
//! the scalar-parameter load (`ld.param.u32` into an integer register), the
//! buffer element type (a 4-byte stride), and `@ZeroDrift`.
//!
//! What they are now: `F64` is a first-class register type (`%fd`, `.f64`
//! arithmetic, an 8-byte stride); `F16` is a buffer element format like the
//! sub-word integers - a load widens it to F32 exactly, a store rounds to
//! nearest-even - and an `F16` VALUE (local, zero-init, parameter) is refused
//! by name, because the LLVM backend's `half` rounds after every operation and
//! a promoted f32 register would not.
//!
//! **Checking the PTX text, or that `ptxas` accepts it, cannot catch either
//! bug** - the wrong module was well-formed. So the device tests run each
//! kernel into a buffer pre-filled with 0xAB and compare the BYTES that come
//! back, including the element beside the one written. The source-level tests
//! below them run without a GPU.

use std::path::Path;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Emitted {
    ok: bool,
    log: String,
    ptx: Option<String>,
}

/// Compiles `src` (a kernel named `k`; `fn main() {}` is appended) with
/// `--emit-ptx`, in its own temp dir: `--emit-ptx` writes next to its input,
/// and a per-test tag alone is not unique across tests in one file.
fn emit(tag: &str, src: &str) -> Emitted {
    static SEQ: AtomicUsize = AtomicUsize::new(0);
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut bin = std::env::current_exe().unwrap();
    bin.pop();
    if bin.ends_with("deps") {
        bin.pop();
    }
    let dir = std::env::temp_dir().join(format!(
        "y_float_widths_{}_{}_{}",
        std::process::id(),
        tag,
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("k.ysu");
    std::fs::write(&file, format!("{}\nfn main() {{}}\n", src)).unwrap();
    let out = Command::new(bin.join("Y"))
        .arg(&file)
        .arg("--emit-ptx")
        .current_dir(repo)
        .output()
        .expect("run Y");
    let log = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let ptx = std::fs::read_to_string(dir.join("k.ptx")).ok();
    let _ = std::fs::remove_dir_all(&dir);
    Emitted { ok: out.status.success(), log, ptx }
}

fn compiled(tag: &str, src: &str) -> String {
    let e = emit(tag, src);
    assert!(e.ok, "{} must compile:\n{}", tag, e.log);
    e.ptx.unwrap_or_else(|| panic!("{} compiled but wrote no .ptx:\n{}", tag, e.log))
}

/// Launches `k` once, single-threaded. `bufs` are the initial contents of the
/// kernel's buffer parameters (in order, before any scalar); `scalars` are the
/// raw bits of the scalar parameters after them. Returns every buffer's bytes
/// after the launch, or `None` with no CUDA driver.
fn run(ptx: &str, bufs: &[Vec<u8>], scalars: &[u64]) -> Option<Vec<Vec<u8>>> {
    use y::cuda_runtime::CudaContext;
    let ctx = CudaContext::new()?;
    let module = ctx.load_ptx(ptx, "k").expect("PTX failed to load");
    let mut dev = Vec::new();
    for init in bufs {
        let b = ctx.alloc(init.len()).unwrap();
        ctx.memcpy_htod_at(&b, 0, init).unwrap();
        dev.push(b);
    }
    let mut args: Vec<u64> = dev.iter().map(|b| b.device_ptr()).collect();
    args.extend_from_slice(scalars);
    ctx.launch(&module, (1, 1, 1), (1, 1, 1), 0, &args)
        .expect("launch failed");
    ctx.synchronize().expect("kernel did not complete");
    Some(
        dev.iter()
            .map(|b| {
                let mut v = vec![0u8; b.len_bytes()];
                ctx.memcpy_dtoh_at(&mut v, b, 0).unwrap();
                v
            })
            .collect(),
    )
}

fn skip(what: &str) {
    eprintln!("SKIP: no CUDA driver - {} was not checked on the device.", what);
}

const POISON: u8 = 0xAB;

fn poisoned(bytes: usize) -> Vec<u8> {
    vec![POISON; bytes]
}

fn f64_at(b: &[u8], i: usize) -> u64 {
    u64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().unwrap())
}

fn f32_at(b: &[u8], i: usize) -> u32 {
    u32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap())
}

fn u16_at(b: &[u8], i: usize) -> u16 {
    u16::from_le_bytes([b[i * 2], b[i * 2 + 1]])
}

const UNTOUCHED16: u16 = 0xABAB;

// ───────────────────────────── on the device ─────────────────────────────

/// The reported program, byte for byte: all eight bytes of element 0 are the
/// double 6.0, and element 1 is still poison.
#[test]
fn an_f64_value_is_computed_and_stored_as_f64() {
    let ptx = compiled(
        "f64_repro",
        "kernel k(Out: GlobalMemory<F64>) {\n    let x: F64 = 3.0;\n    store(Out, x + x);\n}",
    );
    let Some(out) = run(&ptx, &[poisoned(16)], &[]) else {
        return skip("the F64 store");
    };
    let got = &out[0];
    assert_eq!(
        f64_at(got, 0),
        6.0f64.to_bits(),
        "element 0 must hold the double 6.0; the bug stored the FLOAT 6.0 into its low \
         four bytes (-2.53e-98 as a double). Bytes: {:02x?}",
        &got[..8]
    );
    assert_eq!(&got[8..], &[POISON; 8], "element 1 was never written and must still be poison");
}

/// A literal in an F64 context is an F64 literal - the type checker says so -
/// so it must not arrive as the widened f32 rounding: `0.1` would be
/// 0.10000000149011612 and `1.0 / 3.0` would be the f32 quotient,
/// 0.3333333432674408.
#[test]
fn an_f64_literal_keeps_its_double_precision() {
    let ptx = compiled(
        "f64_literals",
        "kernel k(Out: GlobalMemory<F64>) {\n    \
         let a: F64 = 0.1;\n    \
         let b: F64 = 1.0 / 3.0;\n    \
         let c: F64 = -0.1;\n    \
         store(Out[0], a);\n    \
         store(Out[1], b);\n    \
         store(Out[2], c);\n}",
    );
    let Some(out) = run(&ptx, &[poisoned(32)], &[]) else {
        return skip("F64 literal precision");
    };
    let got = &out[0];
    for (i, want, what) in [(0, 0.1f64, "0.1"), (1, 1.0f64 / 3.0, "1.0 / 3.0"), (2, -0.1f64, "-0.1")] {
        assert_eq!(
            f64_at(got, i),
            want.to_bits(),
            "`{}` in an F64 context read back as {} instead of {}",
            what,
            f64::from_bits(f64_at(got, i)),
            want
        );
    }
    assert_eq!(&got[24..], &[POISON; 8], "element 3 was never written");
}

/// A scalar F64 parameter is eight bytes of double. It used to take a `.b32`
/// slot and an `ld.param.u32`, i.e. the host's double read as an integer.
/// `x*x + 0.1` is also the F64 fusion: `ptxas` contracts `mul.f64` + `add.f64`
/// into one `DFMA` (measured), so the emitter states it as `fma.rn.f64` and
/// the answer is the single-rounding `mul_add`.
#[test]
fn an_f64_parameter_arrives_as_eight_bytes() {
    let ptx = compiled(
        "f64_param",
        "kernel k(Out: GlobalMemory<F64>, x: F64) {\n    \
         store(Out[0], x * x + 0.1);\n    \
         store(Out[1], x / 3.0);\n    \
         store(Out[2], x - 1.0);\n}",
    );
    assert!(ptx.contains(".param .f64"), "the parameter slot must be eight bytes:\n{ptx}");
    let x = 1.1f64;
    let Some(out) = run(&ptx, &[poisoned(32)], &[x.to_bits()]) else {
        return skip("the F64 parameter");
    };
    let got = &out[0];
    let want = [x.mul_add(x, 0.1), x / 3.0, x - 1.0];
    for (i, w) in want.iter().enumerate() {
        assert_eq!(
            f64_at(got, i),
            w.to_bits(),
            "element {}: got {} want {}",
            i,
            f64::from_bits(f64_at(got, i)),
            w
        );
    }
    assert_eq!(&got[24..], &[POISON; 8]);
}

/// An `F64` buffer is read at an eight-byte stride. The stride used to be
/// f32's four, so element 1 was the high half of element 0 and the low half
/// of element 1. The masked read past the bound is the double zero.
#[test]
fn an_f64_buffer_is_loaded_at_an_eight_byte_stride() {
    let ptx = compiled(
        "f64_load",
        "kernel k(In: GlobalMemory<F64>, Out: GlobalMemory<F64>) {\n    \
         let v: F64 = block_ptr2d_load(In, 0, 1, 3, 1, 3);\n    \
         let w: F64 = GlobalMemory::load(In[2]);\n    \
         let m: F64 = block_ptr2d_load(In, 0, 7, 3, 1, 3);\n    \
         store(Out[0], v + w);\n    \
         store(Out[1], m + 0.5);\n}",
    );
    let input: Vec<u8> = [1.25f64, 3.5, 7.75].iter().flat_map(|v| v.to_le_bytes()).collect();
    let Some(out) = run(&ptx, &[input, poisoned(24)], &[]) else {
        return skip("the F64 load");
    };
    let got = &out[1];
    assert_eq!(f64_at(got, 0), 11.25f64.to_bits(), "3.5 + 7.75, read at an 8-byte stride");
    assert_eq!(f64_at(got, 1), 0.5f64.to_bits(), "an out-of-bounds masked load must read 0.0");
    assert_eq!(&got[16..], &[POISON; 8]);
}

/// An `F16` slot gets two bytes, rounded to nearest-even, and nothing else.
///
/// * `x + x` with `x = 3.0` is `0x4600`. The bug wrote four bytes here and
///   clobbered the next element.
/// * 2051.0 lies exactly between f16 2050 (`0x6801`) and 2052 (`0x6802`):
///   nearest-EVEN is `0x6802`, truncation would be `0x6801`.
/// * A literal `0.1` is `0x2E66`.
/// * `2051 - 2^-20` as an F64 is just BELOW that tie, so direct rounding gives
///   `0x6801` - but rounding it to f32 first lands exactly ON the tie and then
///   goes to `0x6802`. That is the double rounding `cvt.rn.f16.f64` avoids.
#[test]
fn an_f16_store_rounds_to_nearest_even_and_writes_two_bytes() {
    let ptx = compiled(
        "f16_store",
        "kernel k(Out: GlobalMemory<F16>, d: F64) {\n    \
         let x: F32 = 3.0;\n    \
         let t: F32 = 2051.0;\n    \
         store(Out[1], x + x);\n    \
         store(Out[2], t);\n    \
         store(Out[4], 0.1);\n    \
         store(Out[6], d);\n}",
    );
    let d = 2051.0f64 - 2f64.powi(-20);
    assert_eq!((d as f32) as f64, 2051.0, "fixture: d must round to the tie in f32");
    let Some(out) = run(&ptx, &[poisoned(16)], &[d.to_bits()]) else {
        return skip("the F16 store");
    };
    let got = &out[0];
    let want: [(usize, u16, &str); 8] = [
        (0, UNTOUCHED16, "never written"),
        (1, 0x4600, "3.0 + 3.0"),
        (2, 0x6802, "2051.0, a tie, rounded to EVEN"),
        (3, UNTOUCHED16, "never written"),
        (4, 0x2E66, "the literal 0.1"),
        (5, UNTOUCHED16, "never written"),
        (6, 0x6801, "2051 - 2^-20 rounded ONCE from the double"),
        (7, UNTOUCHED16, "never written"),
    ];
    for (i, w, what) in want {
        assert_eq!(u16_at(got, i), w, "F16 element {} ({}): got {:#06x}", i, what, u16_at(got, i));
    }
}

/// An `F16` load widens the half exactly - it used to load four bytes as an
/// f32 at a 4-byte stride - and a masked read is +0.0.
#[test]
fn an_f16_buffer_is_loaded_as_its_exact_value() {
    let ptx = compiled(
        "f16_load",
        "kernel k(In: GlobalMemory<F16>, Out: GlobalMemory<F32>) {\n    \
         let a: F32 = block_ptr2d_load(In, 0, 1, 4, 1, 4);\n    \
         let b: F32 = GlobalMemory::load(In[3]);\n    \
         let c: F32 = block_ptr2d_load(In, 0, 9, 4, 1, 4);\n    \
         store(Out[0], a);\n    \
         store(Out[1], b);\n    \
         store(Out[2], c + 7.0);\n}",
    );
    // 1.0, 3.0, -5.0, 0.0999755859375
    let halves: [u16; 4] = [0x3C00, 0x4200, 0xC500, 0x2E66];
    let input: Vec<u8> = halves.iter().flat_map(|h| h.to_le_bytes()).collect();
    let Some(out) = run(&ptx, &[input, poisoned(16)], &[]) else {
        return skip("the F16 load");
    };
    let got = &out[1];
    assert_eq!(f32_at(got, 0), 3.0f32.to_bits(), "element 1 of the half buffer is 3.0");
    assert_eq!(f32_at(got, 1), 0.0999755859375f32.to_bits(), "element 3 is the half nearest 0.1");
    assert_eq!(f32_at(got, 2), 7.0f32.to_bits(), "a masked read must be +0.0");
    assert_eq!(f32_at(got, 3), u32::from_le_bytes([POISON; 4]), "element 3 was never written");
}

/// Two `@ZeroDrift` sites the F64 work rewrote, both decided by a register's
/// NAME where they meant its TYPE.
///
/// * The initialiser of a fixed-point accumulator was converted only if its
///   register name began with `%f`, so `let acc: F32 = 5;` - an integer
///   literal, which this backend types by value - started at 0 and the kernel
///   stored 0.0.
/// * A term that was not `%f` was widened with `cvt.rn.f64.s64` whatever its
///   width, and `acc += 3` hands it a 32-bit register, which `ptxas` rejects.
///
/// (An F64 or I32 VARIABLE term is refused by the type checker for an F32
/// accumulator, so the literal is the reachable integer case.)
#[test]
fn zero_drift_takes_integer_literal_initialisers_and_terms() {
    let ptx = compiled(
        "drift_terms",
        "kernel k(Out: GlobalMemory<F32>) {\n    \
         @ZeroDrift\n    \
         @bounds(-1024, 1024)\n    \
         let mut acc: F32 = 5;\n    \
         acc += 3;\n    \
         acc += 0.25;\n    \
         store(Out, acc);\n}",
    );
    let Some(out) = run(&ptx, &[poisoned(8)], &[]) else {
        return skip("the @ZeroDrift integer literals");
    };
    assert_eq!(
        f32_at(&out[0], 0),
        8.25f32.to_bits(),
        "5 (integer initialiser) + 3 (integer term) + 0.25; got {}",
        f32::from_bits(f32_at(&out[0], 0))
    );
    assert_eq!(&out[0][4..], &[POISON; 4]);
}

/// `let x: F32 = {};` emitted `mov.f32 %f0, 0;` - an integer immediate, which
/// a float `mov` refuses - so `ptxas` rejected the module after a clean
/// compile. Loading it on the device is the assembly check.
#[test]
fn a_float_zero_initialiser_assembles_and_is_zero() {
    let ptx = compiled(
        "float_zero",
        "kernel k(Out: GlobalMemory<F32>, D: GlobalMemory<F64>) {\n    \
         let x: F32 = {};\n    \
         let y: F64 = {};\n    \
         store(Out, x + 1.5);\n    \
         store(D, y + 2.5);\n}",
    );
    let Some(out) = run(&ptx, &[poisoned(8), poisoned(16)], &[]) else {
        return skip("the float zero-initialisers");
    };
    assert_eq!(f32_at(&out[0], 0), 1.5f32.to_bits());
    assert_eq!(f64_at(&out[1], 0), 2.5f64.to_bits());
}

/// A `type X = F64;` alias is honoured. The PTX backend wrote aliases as a
/// comment and nothing else, so `let x: Dbl` read as "no annotation". (Not
/// `D`: a single capital A-D lexes as an MMA fragment role, not a name.)
#[test]
fn a_type_alias_to_f64_is_honoured() {
    let ptx = compiled(
        "f64_alias",
        "kernel k(Out: GlobalMemory<F64>) {\n    \
         type Dbl = F64;\n    \
         let x: Dbl = 0.1;\n    \
         store(Out, x + x);\n}",
    );
    assert!(ptx.contains("add.f64"), "the alias must lower as F64:\n{ptx}");
    let Some(out) = run(&ptx, &[poisoned(16)], &[]) else {
        return skip("the F64 alias");
    };
    assert_eq!(f64_at(&out[0], 0), (0.1f64 + 0.1).to_bits());
}

// ───────────────────────────── controls ─────────────────────────────

/// The F32 path is untouched: same instructions as before (no `%fd`, no
/// `.f64`), and on the device 6.0f in four bytes with the next four poison.
#[test]
fn the_f32_path_is_unchanged() {
    let ptx = compiled(
        "f32_control",
        "kernel k(Out: GlobalMemory<F32>) {\n    let x: F32 = 3.0;\n    store(Out, x + x);\n}",
    );
    for needle in ["mov.f32 %f0, 3.0;", "add.f32 %f1, %f0, %f0;", "st.global.f32 [%rd0], %f1;"] {
        assert!(ptx.contains(needle), "the F32 lowering moved - missing `{}`:\n{}", needle, ptx);
    }
    assert!(!ptx.contains("%fd") && !ptx.contains(".f64"), "an F32 kernel grew doubles:\n{ptx}");
    let Some(out) = run(&ptx, &[poisoned(8)], &[]) else {
        return skip("the F32 control");
    };
    assert_eq!(f32_at(&out[0], 0), 6.0f32.to_bits());
    assert_eq!(&out[0][4..], &[POISON; 4]);
}

/// Integer stores still convert into the buffer's element type and still
/// write only their own slot.
#[test]
fn integer_stores_are_unchanged() {
    let ptx = compiled(
        "int_control",
        "kernel k(A: GlobalMemory<U32>, B: GlobalMemory<U64>) {\n    \
         store(A[1], 7);\n    \
         store(B, 9);\n}",
    );
    let Some(out) = run(&ptx, &[poisoned(12), poisoned(16)], &[]) else {
        return skip("the integer control");
    };
    assert_eq!(f32_at(&out[0], 0), u32::from_le_bytes([POISON; 4]));
    assert_eq!(f32_at(&out[0], 1), 7);
    assert_eq!(f32_at(&out[0], 2), u32::from_le_bytes([POISON; 4]));
    // The BARE-buffer form into a U64 wrote `st.global.u32` - four bytes of
    // an eight-byte slot - because only `Expr::Index` consulted the buffer.
    assert_eq!(f64_at(&out[1], 0), 9, "a bare-buffer store into U64 must write all eight bytes");
    assert_eq!(&out[1][8..], &[POISON; 8]);
}

// ───────────────────────────── refusals ─────────────────────────────

fn refused(tag: &str, src: &str, needles: &[&str]) {
    let e = emit(tag, src);
    assert!(!e.ok, "{} must be refused:\n{}", tag, e.log);
    assert!(e.ptx.is_none(), "{} was refused but still wrote a .ptx:\n{}", tag, e.log);
    for n in needles {
        assert!(e.log.contains(n), "{}: the refusal must say `{}`:\n{}", tag, n, e.log);
    }
}

/// An `F16` VALUE is refused by name at every site a value is declared - the
/// reported program first. It used to compile as f32.
#[test]
fn an_f16_value_is_refused_by_name() {
    let why = "buffer element type in this backend, not a value type";
    refused(
        "f16_let",
        "kernel k(Out: GlobalMemory<F16>) {\n    let x: F16 = 3.0;\n    store(Out, x + x);\n}",
        &["let x: F16", why],
    );
    refused(
        "f16_zero",
        "kernel k(Out: GlobalMemory<F32>) {\n    let x: F16 = {};\n    store(Out, x);\n}",
        &["let x: F16", why],
    );
    refused(
        "f16_param",
        "kernel k(Out: GlobalMemory<F32>, h: F16) {\n    store(Out, h);\n}",
        &["parameter `h: F16`", why],
    );
    refused(
        "f16_alias",
        "kernel k(Out: GlobalMemory<F32>) {\n    type Half = F16;\n    let x: Half = 1.0;\n    store(Out, x);\n}",
        &[why],
    );
}

/// The other annotations `from_name` did not know and every caller ignored.
#[test]
fn other_unlowerable_value_types_are_refused() {
    refused(
        "q_let",
        "kernel k(Out: GlobalMemory<F32>) {\n    let q: Q16.16 = 1.5;\n    store(Out, q);\n}",
        &["Q16.16", "fixed-point"],
    );
    refused(
        "q_param",
        "kernel k(Out: GlobalMemory<F32>, q: Q16.16) {\n    store(Out, 1.0);\n}",
        &["q: Q16.16"],
    );
    // A buffer ELEMENT type the backend cannot stride recorded no element
    // type at all, which every load and store reads as f32: HEAD wrote
    // `store(Out[1], 1)` into a `GlobalMemory<Q16.16>` as the raw u32 1, not
    // the fixed-point 1.0, at a 4-byte stride - under a clean compile.
    for elem in ["Q16.16", "bool"] {
        refused(
            "elem_unknown",
            &format!("kernel k(Out: GlobalMemory<{}>) {{\n    store(Out[1], 1);\n}}", elem),
            &[&format!("GlobalMemory<{}>", elem), "4-byte stride"],
        );
    }
}

/// Every intrinsic that hardcodes `.f32` at a 4-byte stride refuses an F16 or
/// F64 buffer. Only two of the eleven checked the buffer at all, and they
/// refused integers only - an F16/F64 buffer had no recorded element type, so
/// it read as f32.
#[test]
fn f32_only_intrinsics_refuse_f16_and_f64_buffers() {
    for (tag, sig, body, needle) in [
        ("tile_f64", "A: GlobalMemory<F64>", "let v: F32 = block_tile_load(A, 0, 4);", "F64 buffer"),
        ("tile_f16", "A: GlobalMemory<F16>", "let v: F32 = block_tile_load(A, 0, 4);", "F16 buffer"),
        ("bt_f64", "A: GlobalMemory<F64>", "let v: F32 = BlockTile::load(A, 0, 4);", "F64 buffer"),
        ("ldv4_f16", "A: GlobalMemory<F16>", "let v: F32 = load_v4(A);", "F16 buffer"),
        ("gmv4_f64", "A: GlobalMemory<F64>", "let v: F32 = GlobalMemory::load_v4(A[0]);", "F64 buffer"),
        ("p3d_f16", "A: GlobalMemory<F16>", "let v: F32 = block_ptr3d_load(A, 0, 0, 0);", "F16 buffer"),
        ("v4_2d_f16", "A: GlobalMemory<F16>", "let v: U32x4 = block_ptr2d_load_v4(A, 0, 0, 4, 1, 4);", "no F16 form"),
        ("v4_2d_f64", "A: GlobalMemory<F64>", "let v: U32x4 = block_ptr2d_load_v4(A, 0, 0, 4, 1, 4);", "only 32-bit"),
    ] {
        refused(tag, &format!("kernel k({}) {{\n    {}\n}}", sig, body), &[needle]);
    }
}

/// The control for the refusals: the same intrinsics on an F32 buffer still
/// compile, so the check is on the element type and not on the intrinsic.
#[test]
fn f32_only_intrinsics_still_accept_f32_buffers() {
    for body in [
        "let v: F32 = block_tile_load(A, 0, 4);",
        "let v: F32 = BlockTile::load(A, 0, 4);",
        "let v: F32 = load_v4(A);",
        "let v: F32 = block_ptr3d_load(A, 0, 0, 0);",
    ] {
        let e = emit("f32_ok", &format!("kernel k(A: GlobalMemory<F32>) {{\n    {}\n}}", body));
        assert!(e.ok && e.ptx.is_some(), "`{}` on an F32 buffer must compile:\n{}", body, e.log);
    }
}

/// `A[i] = v` computed the value and DROPPED the store - the assignment arm
/// handled only a plain name - under a clean compile and exit 0. It is lowered
/// now as exactly `store(A[i], v)`: the element's stride and type, including an
/// F16 slot rounded and an eight-byte U64 one, and nothing beside it.
#[test]
fn an_indexed_assignment_stores_what_it_names() {
    let ptx = compiled(
        "idx_assign",
        "kernel k(F: GlobalMemory<F32>, H: GlobalMemory<F16>, W: GlobalMemory<U64>) {\n    \
         let i: I32 = 1;\n    \
         F[i] = 2.5;\n    \
         H[i] = 3.0;\n    \
         W[i] = 7;\n}",
    );
    let Some(out) = run(&ptx, &[poisoned(12), poisoned(6), poisoned(24)], &[]) else {
        return skip("indexed assignment");
    };
    let p32 = u32::from_le_bytes([POISON; 4]);
    assert_eq!([f32_at(&out[0], 0), f32_at(&out[0], 1), f32_at(&out[0], 2)], [p32, 2.5f32.to_bits(), p32]);
    assert_eq!([u16_at(&out[1], 0), u16_at(&out[1], 1), u16_at(&out[1], 2)], [UNTOUCHED16, 0x4200, UNTOUCHED16]);
    assert_eq!(f64_at(&out[2], 1), 7, "all eight bytes of the U64 element");
    assert_eq!(&out[2][..8], &[POISON; 8]);
    assert_eq!(&out[2][16..], &[POISON; 8]);
}

/// `A[i] += v` is `A[i] = A[i] + v`, and READING `A[i]` evaluates to the
/// element's address in this backend - so it is refused rather than storing
/// the address plus `v`. A target that is not a name or a buffer element has
/// no storage here and is refused too.
#[test]
fn assignments_this_backend_cannot_honour_are_refused() {
    refused(
        "idx_compound",
        "kernel k(Out: GlobalMemory<F32>) {\n    Out[0] += 2.5;\n}",
        &["compound assignment to an element", "ADDRESS"],
    );
}
