//! Artifact identity for the offline-validated `exact_pv` cubin.
//!
//! A receipt is produced only after successful translation validation by
//! `tools/ptxas_tval/exact_pv_artifact.py`. This is a trusted local build
//! receipt, not an authenticated proof: its producer and initial storage
//! remain trusted, as do the existing validator, disassembler and ISA model.
//! Opening pins the receipt, and every load rechecks all artifact hashes.
//! A fixed full-PTX subject pin also limits this API to the transcription
//! reviewed against ExactPvExact.v; that transcription remains trusted.
//! The exact checked in-memory cubin bytes are handed to the CUDA driver.

use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

const RECEIPT: &str = "receipt.txt";
const KEYS: [&str; 10] = [
    "format",
    "entry",
    "target",
    "optimization",
    "validator",
    "verdict",
    "obligations",
    "ptx_sha256",
    "sass_sha256",
    "cubin_sha256",
];

/// A nonempty launch domain for the fixed reviewed exact_pv subject.
///
/// These checks imply the arithmetic theorem's input-index and accumulator
/// premises, and the separate output-index licence. They do not prove the
/// model-to-PTX transcription or hardware execution. T = 0 is deliberately
/// rejected: ExactPvExact's current capstone requires T >= 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExactPvShape {
    batch: i32,
    queries: i32,
    keys: i32,
    depth: i32,
    elements: [i32; 3],
    bytes: [usize; 3],
}

impl ExactPvShape {
    pub fn new(batch: i64, queries: i64, keys: i64, depth: i64) -> Result<Self, String> {
        let mut dims = [0i32; 4];
        for (i, (name, value)) in [("B", batch), ("Q", queries), ("T", keys), ("D", depth)]
            .into_iter().enumerate()
        {
            if value <= 0 || value > i32::MAX as i64 {
                return Err(format!("checked exact_pv requires 1 <= {name} <= i32::MAX"));
            }
            dims[i] = value as i32;
        }
        // Compute outside the parameter width before narrowing any product.
        let (b, q, t, d) = (batch as u128, queries as u128, keys as u128, depth as u128);
        let products = [b * q * t, b * t * d, b * q * d];
        let mut elements = [0i32; 3];
        let mut bytes = [0usize; 3];
        for (i, count) in products.into_iter().enumerate() {
            if count > i32::MAX as u128 {
                return Err(format!("checked exact_pv {} element count exceeds i32::MAX",
                    ["P", "V", "Out"][i]));
            }
            elements[i] = count as i32;
            bytes[i] = usize::try_from(count * [4u128, 1, 8][i])
                .map_err(|_| "checked exact_pv byte extent exceeds host size width")?;
        }
        if t * (u32::MAX as u128) * 128 > i64::MAX as u128 {
            return Err("checked exact_pv contraction exceeds the full-domain i64 accumulator licence".into());
        }
        Ok(Self { batch: dims[0], queries: dims[1], keys: dims[2], depth: dims[3], elements, bytes })
    }

    /// B, Q, T, D in the mathematical source's order.
    pub fn dimensions(self) -> [i32; 4] {
        [self.batch, self.queries, self.keys, self.depth]
    }

    /// P:u32, V:i8, Out:i64 element extents passed to the masked accesses.
    pub fn element_counts(self) -> [i32; 3] { self.elements }

    pub fn required_bytes(self) -> [usize; 3] { self.bytes }

    pub fn grid(self) -> (u32, u32, u32) { (self.queries as u32, self.batch as u32, 1) }

    /// The subject has no d < D guard: rounding this block size up is unsafe.
    pub fn block(self) -> (u32, u32, u32) { (self.depth as u32, 1, 1) }

    pub(crate) fn scalar_parameters(self) -> [i32; 6] {
        [self.keys, self.depth, self.queries, self.elements[0], self.elements[1], self.elements[2]]
    }
}

/// Check live byte spans after ownership has established real allocations.
/// P and V are read-only and may alias; output must not overlap either input.
pub(crate) fn check_launch_buffers(shape: ExactPvShape, buffers: [(u64, usize); 3]) -> Result<(), String> {
    let mut spans = [(0u64, 0u64); 3];
    for (i, (address, available)) in buffers.into_iter().enumerate() {
        let name = ["P", "V", "Out"][i];
        if address == 0 || address % [4u64, 1, 8][i] != 0 {
            return Err(format!("checked exact_pv {name} pointer is null or misaligned"));
        }
        if available < shape.bytes[i] {
            return Err(format!("checked exact_pv {name} allocation is shorter than its live byte extent"));
        }
        let extent = u64::try_from(shape.bytes[i])
            .map_err(|_| "checked exact_pv byte extent exceeds device address width")?;
        let end = address.checked_add(extent)
            .ok_or_else(|| format!("checked exact_pv {name} live address range wraps"))?;
        spans[i] = (address, end);
    }
    let out = spans[2];
    for (i, input) in spans[..2].iter().enumerate() {
        if input.0 < out.1 && out.0 < input.1 {
            return Err(format!("checked exact_pv Out overlaps the live {} input", ["P", "V"][i]));
        }
    }
    Ok(())
}

/// A validated artifact bundle whose identity cannot be changed by its caller.
///
/// Use `CudaContext::load_checked_exact_pv` and `launch_checked_exact_pv` for
/// execution inside the checked domain. `load_verified_exact_pv` checks the
/// artifact alone; ordinary `load_ptx` conveys neither claim.
#[derive(Debug)]
pub struct ValidatedExactPv {
    directory: PathBuf,
    receipt_bytes: Vec<u8>,
    ptx_sha256: String,
    sass_sha256: String,
    cubin_sha256: String,
}

impl ValidatedExactPv {
    /// Open a successful trusted build receipt and check its artifact hashes.
    /// `expected_ptx_sha256` additionally binds the bundle to a caller's source
    /// compilation, rather than accepting any locally validated exact_pv PTX.
    pub fn open(
        bundle_dir: impl AsRef<Path>,
        expected_ptx_sha256: Option<&str>,
    ) -> Result<Self, String> {
        let directory = fs::canonicalize(bundle_dir.as_ref())
            .map_err(|e| format!("cannot identify exact_pv bundle: {e}"))?;
        let receipt_bytes = read(&directory, RECEIPT)?;
        let receipt = parse_receipt(&receipt_bytes)?;
        if let Some(expected) = expected_ptx_sha256 {
            if !is_digest(expected) || expected != receipt["ptx_sha256"] {
                return Err("exact_pv PTX does not match the expected SHA-256".into());
            }
        }
        let artifact = Self {
            directory,
            ptx_sha256: receipt["ptx_sha256"].to_owned(),
            sass_sha256: receipt["sass_sha256"].to_owned(),
            cubin_sha256: receipt["cubin_sha256"].to_owned(),
            receipt_bytes,
        };
        artifact.checked_cubin()?;
        Ok(artifact)
    }

    pub fn cubin_sha256(&self) -> &str {
        &self.cubin_sha256
    }

    /// Read once, check once, and return the same owned bytes for loading.
    /// No path is reopened by the driver, closing a hash/load file race.
    pub(crate) fn checked_cubin(&self) -> Result<Vec<u8>, String> {
        if read(&self.directory, RECEIPT)? != self.receipt_bytes {
            return Err("exact_pv validation receipt changed after opening".into());
        }
        for (file, digest) in [
            ("exact_pv.ptx", &self.ptx_sha256),
            ("exact_pv.sass", &self.sass_sha256),
        ] {
            let bytes = read(&self.directory, file)?;
            check_digest(file, &bytes, digest)?;
            if file == "exact_pv.ptx" {
                require_proved_subject(&bytes)?;
            }
        }
        let cubin = read(&self.directory, "exact_pv.cubin")?;
        check_digest("exact_pv.cubin", &cubin, &self.cubin_sha256)?;
        identify_cubin(&cubin)?;
        Ok(cubin)
    }
}

/// Full subject identity reviewed against ExactPvExact's model. This closes
/// the opcode-presence gate without claiming a formally verified PTX decoder.
fn require_proved_subject(bytes: &[u8]) -> Result<(), String> {
    let mut clean = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i..].starts_with(b"//") {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            clean.push(b' ');
        } else if bytes[i..].starts_with(b"/*") {
            i += 2;
            while i < bytes.len() && !bytes[i..].starts_with(b"*/") {
                i += 1;
            }
            if i == bytes.len() {
                return Err("unterminated exact_pv PTX comment".into());
            }
            i += 2;
            clean.push(b' ');
        } else {
            // The reviewed subject contains no quoted directives or non-ASCII
            // tokens. Refuse them rather than parsing a different language.
            if !bytes[i].is_ascii() || bytes[i] == b'"' || bytes[i] == 0 {
                return Err("unreviewed exact_pv PTX token".into());
            }
            clean.push(bytes[i]);
            i += 1;
        }
    }
    let text = std::str::from_utf8(&clean).map_err(|_| "invalid exact_pv PTX")?;
    let canonical = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let expected = include_str!("../tools/ptxas_tval/exact_pv_subject.sha256").trim();
    if format!("{:x}", Sha256::digest(canonical.as_bytes())) != expected {
        return Err("PTX differs from the reviewed ExactPvExact proof subject".into());
    }
    Ok(())
}

fn read(directory: &Path, name: &str) -> Result<Vec<u8>, String> {
    fs::read(directory.join(name)).map_err(|e| format!("cannot read exact_pv {name}: {e}"))
}

fn is_digest(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn check_digest(name: &str, bytes: &[u8], expected: &str) -> Result<(), String> {
    let actual = format!("{:x}", Sha256::digest(bytes));
    if actual != expected {
        return Err(format!("exact_pv {name} SHA-256 mismatch"));
    }
    Ok(())
}

fn parse_receipt(bytes: &[u8]) -> Result<BTreeMap<&str, &str>, String> {
    let text = std::str::from_utf8(bytes).map_err(|_| "exact_pv receipt is not UTF-8")?;
    let mut fields = BTreeMap::new();
    for line in text.lines() {
        let (key, value) = line
            .split_once('=')
            .ok_or("malformed exact_pv receipt line")?;
        if !KEYS.contains(&key) || fields.insert(key, value).is_some() {
            return Err("unknown or duplicate exact_pv receipt field".into());
        }
    }
    if fields.len() != KEYS.len() {
        return Err("incomplete exact_pv validation receipt".into());
    }
    for (key, expected) in [
        ("format", "y-exact-pv-cubin-v1"),
        ("entry", "exact_pv"),
        ("target", "sm_89"),
        ("optimization", "1"),
        ("validator", "loopval"),
        ("verdict", "VALIDATED"),
    ] {
        if fields[key] != expected {
            return Err(format!(
                "unsupported or unsuccessful exact_pv receipt: {key}"
            ));
        }
    }
    let obligations = fields["obligations"];
    if obligations.starts_with('0')
        || !obligations.bytes().all(|b| b.is_ascii_digit())
        || obligations.parse::<u64>().ok().filter(|&n| n > 0).is_none()
    {
        return Err("exact_pv receipt has no validated obligations".into());
    }
    for key in ["ptx_sha256", "sass_sha256", "cubin_sha256"] {
        if !is_digest(fields[key]) {
            return Err(format!("invalid exact_pv receipt digest: {key}"));
        }
    }
    Ok(fields)
}

fn slice(bytes: &[u8], offset: u64, size: u64) -> Result<&[u8], String> {
    let start = usize::try_from(offset).map_err(|_| "cubin offset exceeds host range")?;
    let end = offset.checked_add(size).ok_or("cubin extent overflow")?;
    let end = usize::try_from(end).map_err(|_| "cubin extent exceeds host range")?;
    bytes
        .get(start..end)
        .ok_or_else(|| "truncated exact_pv cubin".into())
}

fn u16_at(bytes: &[u8], offset: u64) -> Result<u16, String> {
    Ok(u16::from_le_bytes(
        slice(bytes, offset, 2)?.try_into().unwrap(),
    ))
}
fn u32_at(bytes: &[u8], offset: u64) -> Result<u32, String> {
    Ok(u32::from_le_bytes(
        slice(bytes, offset, 4)?.try_into().unwrap(),
    ))
}
fn u64_at(bytes: &[u8], offset: u64) -> Result<u64, String> {
    Ok(u64::from_le_bytes(
        slice(bytes, offset, 8)?.try_into().unwrap(),
    ))
}

/// Restrict identification to the CUDA ELF64 format and sm_89 target actually
/// validated here. Unknown encodings fail closed; this is not a general ELF
/// parser. The driver remains responsible for loading a structurally valid ELF.
fn identify_cubin(bytes: &[u8]) -> Result<(), String> {
    if slice(bytes, 0, 7)? != b"\x7fELF\x02\x01\x01"
        || slice(bytes, 7, 2)? != [0x41, 8]
        || u16_at(bytes, 16)? != 2
        || u16_at(bytes, 18)? != 190
        || u32_at(bytes, 20)? != 1
        || u16_at(bytes, 52)? != 64
    {
        return Err("exact_pv artifact is not a CUDA ELF64 executable".into());
    }
    // ptxas sm_89 ELF flags: CUDA encoding version 6, SM 89, 64-bit ABI.
    // Reject future/unknown encodings instead of guessing the target.
    if u32_at(bytes, 48)? != 0x0600_5904 {
        return Err("exact_pv cubin target/flags are not the validated sm_89 format".into());
    }
    let program_count = u16_at(bytes, 56)? as u64;
    if program_count != 0 {
        if u16_at(bytes, 54)? != 56 {
            return Err("cannot identify cubin program headers".into());
        }
        slice(bytes, u64_at(bytes, 32)?, 56 * program_count)?;
    }
    let section_offset = u64_at(bytes, 40)?;
    let section_size = u16_at(bytes, 58)? as u64;
    let section_count = u16_at(bytes, 60)? as u64;
    let string_index = u16_at(bytes, 62)? as u64;
    if section_size != 64
        || section_count == 0
        || string_index == 0
        || string_index >= section_count
    {
        return Err("cannot identify exact_pv cubin sections".into());
    }
    slice(bytes, section_offset, section_size * section_count)?;
    let strings_header = section_offset + string_index * section_size;
    if u32_at(bytes, strings_header + 4)? != 3 {
        return Err("cubin section names are not a string table".into());
    }
    let strings = slice(
        bytes,
        u64_at(bytes, strings_header + 24)?,
        u64_at(bytes, strings_header + 32)?,
    )?;
    let mut found = 0;
    for i in 0..section_count {
        let header = section_offset + i * section_size;
        let name_offset = u32_at(bytes, header)? as usize;
        let name_bytes = strings
            .get(name_offset..)
            .ok_or("cubin section name out of bounds")?;
        let name_end = name_bytes
            .iter()
            .position(|&b| b == 0)
            .ok_or("unterminated cubin section name")?;
        if name_bytes[..name_end].starts_with(b".text.")
            && &name_bytes[..name_end] != b".text.exact_pv"
        {
            return Err("cubin contains an unvalidated text entry".into());
        }
        if &name_bytes[..name_end] == b".text.exact_pv" {
            let flags = u64_at(bytes, header + 8)?;
            let size = u64_at(bytes, header + 32)?;
            if u32_at(bytes, header + 4)? != 1 || flags & 6 != 6 || size == 0 {
                return Err("exact_pv ELF section is not executable code".into());
            }
            slice(bytes, u64_at(bytes, header + 24)?, size)?;
            found += 1;
        }
    }
    if found != 1 {
        return Err("cubin must identify exactly one .text.exact_pv entry".into());
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    pub(crate) struct Bundle {
        pub directory: PathBuf,
        pub cubin: Vec<u8>,
    }

    impl Bundle {
        pub(crate) fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let directory = std::env::temp_dir().join(format!(
                "y-exact-pv-artifact-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed),
            ));
            fs::create_dir(&directory).unwrap();
            // A minimal ELF fixture for artifact identification, not GPU code.
            let names = b"\0.shstrtab\0.text.exact_pv\0";
            let text_offset = 64 + 3 * 64 + names.len();
            let mut cubin = vec![0; text_offset + 16];
            cubin[..9].copy_from_slice(b"\x7fELF\x02\x01\x01\x41\x08");
            cubin[16..18].copy_from_slice(&2u16.to_le_bytes());
            cubin[18..20].copy_from_slice(&190u16.to_le_bytes());
            cubin[20..24].copy_from_slice(&1u32.to_le_bytes());
            cubin[40..48].copy_from_slice(&64u64.to_le_bytes());
            cubin[48..52].copy_from_slice(&0x0600_5904u32.to_le_bytes());
            cubin[52..54].copy_from_slice(&64u16.to_le_bytes());
            cubin[58..60].copy_from_slice(&64u16.to_le_bytes());
            cubin[60..62].copy_from_slice(&3u16.to_le_bytes());
            cubin[62..64].copy_from_slice(&1u16.to_le_bytes());
            let strings = 128;
            cubin[strings..strings + 4].copy_from_slice(&1u32.to_le_bytes());
            cubin[strings + 4..strings + 8].copy_from_slice(&3u32.to_le_bytes());
            cubin[strings + 24..strings + 32].copy_from_slice(&256u64.to_le_bytes());
            cubin[strings + 32..strings + 40].copy_from_slice(&(names.len() as u64).to_le_bytes());
            let text = 192;
            cubin[text..text + 4].copy_from_slice(&11u32.to_le_bytes());
            cubin[text + 4..text + 8].copy_from_slice(&1u32.to_le_bytes());
            cubin[text + 8..text + 16].copy_from_slice(&6u64.to_le_bytes());
            cubin[text + 24..text + 32].copy_from_slice(&(text_offset as u64).to_le_bytes());
            cubin[text + 32..text + 40].copy_from_slice(&16u64.to_le_bytes());
            cubin[256..text_offset].copy_from_slice(names);
            fs::write(
                directory.join("exact_pv.ptx"),
                include_bytes!("../tests/exact_pv.ptx"),
            )
            .unwrap();
            fs::write(directory.join("exact_pv.sass"), b"fixture SASS").unwrap();
            fs::write(directory.join("exact_pv.cubin"), &cubin).unwrap();
            let bundle = Self { directory, cubin };
            bundle.write_receipt("VALIDATED");
            bundle
        }

        pub(crate) fn write_receipt(&self, verdict: &str) {
            let mut receipt = format!(
                "format=y-exact-pv-cubin-v1\nentry=exact_pv\ntarget=sm_89\noptimization=1\nvalidator=loopval\nverdict={verdict}\nobligations=14\n"
            );
            for extension in ["ptx", "sass", "cubin"] {
                let bytes = fs::read(self.directory.join(format!("exact_pv.{extension}"))).unwrap();
                receipt.push_str(&format!("{extension}_sha256={:x}\n", Sha256::digest(bytes)));
            }
            fs::write(self.directory.join(RECEIPT), receipt).unwrap();
        }

        pub(crate) fn open(&self) -> ValidatedExactPv {
            ValidatedExactPv::open(&self.directory, None).unwrap()
        }
    }

    impl Drop for Bundle {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.directory);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_support::Bundle;

    #[test]
    fn verified_exact_pv_checks_and_pins_all_artifacts() {
        let bundle = Bundle::new();
        let artifact = bundle.open();
        assert_eq!(artifact.checked_cubin().unwrap(), bundle.cubin);
        assert_eq!(
            artifact.cubin_sha256(),
            format!("{:x}", Sha256::digest(&bundle.cubin))
        );
        let expected = format!(
            "{:x}",
            Sha256::digest(include_bytes!("../tests/exact_pv.ptx"))
        );
        ValidatedExactPv::open(&bundle.directory, Some(&expected)).unwrap();
        assert!(ValidatedExactPv::open(&bundle.directory, Some(&"0".repeat(64))).is_err());
        for extension in ["ptx", "sass", "cubin"] {
            let path = bundle.directory.join(format!("exact_pv.{extension}"));
            let original = fs::read(&path).unwrap();
            let mut changed = original.clone();
            let last = changed.len() - 1;
            changed[last] ^= 1;
            fs::write(&path, changed).unwrap();
            assert!(
                artifact.checked_cubin().is_err(),
                "changed {extension} accepted"
            );
            assert!(ValidatedExactPv::open(&bundle.directory, None).is_err());
            fs::write(&path, original).unwrap();
        }
        // Rewriting the receipt along with a replaced cubin cannot repin an
        // already-open artifact.
        let mut changed = bundle.cubin.clone();
        *changed.last_mut().unwrap() ^= 1;
        fs::write(bundle.directory.join("exact_pv.cubin"), changed).unwrap();
        bundle.write_receipt("VALIDATED");
        assert!(artifact
            .checked_cubin()
            .unwrap_err()
            .contains("receipt changed"));
    }

    #[test]
    fn verified_exact_pv_fails_closed_for_receipts_and_missing_artifacts() {
        let bundle = Bundle::new();
        let good = fs::read_to_string(bundle.directory.join(RECEIPT)).unwrap();
        for receipt in [
            good.replace("verdict=VALIDATED", "verdict=REFUSED"),
            good.replace("obligations=14", "obligations=0"),
            good.replace("obligations=14", "obligations=+14"),
            good.replace("target=sm_89", "target=sm_90"),
            good.replace("entry=exact_pv\n", ""),
            format!("{good}entry=exact_pv\n"),
            format!("{good}unidentified=1\n"),
        ] {
            fs::write(bundle.directory.join(RECEIPT), receipt).unwrap();
            assert!(ValidatedExactPv::open(&bundle.directory, None).is_err());
        }
        fs::write(bundle.directory.join(RECEIPT), &good).unwrap();
        let artifact = bundle.open();
        for file in [RECEIPT, "exact_pv.ptx", "exact_pv.sass", "exact_pv.cubin"] {
            let path = bundle.directory.join(file);
            let saved = fs::read(&path).unwrap();
            fs::remove_file(&path).unwrap();
            assert!(artifact.checked_cubin().is_err());
            assert!(ValidatedExactPv::open(&bundle.directory, None).is_err());
            fs::write(&path, saved).unwrap();
        }
    }

    #[test]
    fn verified_exact_pv_rejects_wrong_index_even_with_a_fresh_valid_receipt() {
        let bundle = Bundle::new();
        let subject = include_str!("../tests/exact_pv.ptx");
        let altered = subject.replacen("add.s32 %r17, %r11, %r14;", "mov.u32 %r17, %r11;", 1);
        assert_ne!(altered, subject, "reviewed index anchor moved");
        fs::write(bundle.directory.join("exact_pv.ptx"), altered).unwrap();
        bundle.write_receipt("VALIDATED");
        assert!(ValidatedExactPv::open(&bundle.directory, None)
            .unwrap_err()
            .contains("reviewed ExactPvExact proof subject"));
    }

    #[test]
    fn verified_exact_pv_subject_identity_ignores_comments_and_whitespace() {
        let subject = include_str!("../tests/exact_pv.ptx");
        let annotated = format!(
            "/* comments only */\n{}",
            subject.replace(
                "    add.s32 %r17, %r11, %r14;",
                "\tadd.s32  %r17, %r11, %r14; // index\n"
            )
        );
        require_proved_subject(annotated.as_bytes()).unwrap();
        assert!(require_proved_subject(format!("{subject}/* unterminated").as_bytes()).is_err());
    }

    #[test]
    fn verified_exact_pv_rejects_unidentified_or_incompatible_elf() {
        let bundle = Bundle::new();
        for index in [
            0, 4, 5, 7, 8, 16, 18, 20, 48, 49, 51, 52, 58, 60, 62, 192, 196,
        ] {
            let mut changed = bundle.cubin.clone();
            changed[index] ^= 1;
            fs::write(bundle.directory.join("exact_pv.cubin"), changed).unwrap();
            bundle.write_receipt("VALIDATED");
            assert!(
                ValidatedExactPv::open(&bundle.directory, None).is_err(),
                "accepted ELF mutation at {index}"
            );
        }
        let mut changed = bundle.cubin.clone();
        changed[40..48].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(identify_cubin(&changed).is_err());
        for length in 0..bundle.cubin.len() {
            assert!(identify_cubin(&bundle.cubin[..length]).is_err());
        }
    }
}
