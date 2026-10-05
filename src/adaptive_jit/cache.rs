//! Local settings and diagnostics only: no executable code is persisted.
//! The checksum detects accidental corruption; it does not authenticate a
//! record against a writer that controls this application-owned directory.

use crate::autotuner::AutotuneCandidate;
use crate::empirical_autotune::LaunchConfig;
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

const FORMAT: &str = "y-adaptive-jit-v1";
const MAX_RECORD_BYTES: u64 = 8192;
static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, PartialEq)]
pub(super) struct CacheRecord {
    pub key: String,
    pub candidate: AutotuneCandidate,
    pub baseline_hash: String,
    pub selected_hash: String,
    pub launch: LaunchConfig,
    pub baseline_us: f64,
    pub selected_us: f64,
    pub candidates_measured: usize,
    pub promoted: bool,
}

fn is_hash(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate(record: &CacheRecord) -> Result<(), String> {
    if !is_hash(&record.key) || !is_hash(&record.baseline_hash) || !is_hash(&record.selected_hash) {
        return Err("adaptive cache requires lowercase SHA-256 identities".into());
    }
    if !record.baseline_us.is_finite()
        || !record.selected_us.is_finite()
        || record.baseline_us <= 0.0
        || record.selected_us <= 0.0
        || record.selected_us > record.baseline_us
        || record.promoted != (record.selected_us < record.baseline_us)
        || !(1..=64).contains(&record.candidates_measured)
    {
        return Err("invalid adaptive cache timing or verdict".into());
    }
    let candidate = &record.candidate;
    if [candidate.cta_m, candidate.cta_n, candidate.cta_k]
        .iter()
        .any(|axis| !(16..=1024).contains(axis) || axis % 16 != 0)
        || !(1..=32).contains(&candidate.warps_m)
        || !(1..=32).contains(&candidate.warps_n)
        || !(1..=32).contains(&candidate.num_warps)
        || !(1..=16).contains(&candidate.num_stages)
        || candidate.warps_m * candidate.warps_n != candidate.num_warps
        || candidate.cta_m % (candidate.warps_m * 16) != 0
        || candidate.cta_n % (candidate.warps_n * 16) != 0
    {
        return Err("invalid adaptive cache candidate".into());
    }
    let launch = record.launch;
    if !(1..=16384).contains(&launch.grid_x)
        || !(1..=16384).contains(&launch.grid_y)
        || launch.threads != candidate.num_warps * 32
        || launch.dyn_smem_bytes > 1024 * 1024
    {
        return Err("invalid adaptive cache launch geometry".into());
    }
    Ok(())
}

fn payload(record: &CacheRecord) -> String {
    let candidate = &record.candidate;
    let launch = record.launch;
    format!(
        "format={FORMAT}\nkey={}\ncta_m={}\ncta_n={}\ncta_k={}\nwarps_m={}\nwarps_n={}\nnum_stages={}\nnum_warps={}\nbaseline_hash={}\nselected_hash={}\ngrid_x={}\ngrid_y={}\nthreads={}\ndyn_smem_bytes={}\nbaseline_us={}\nselected_us={}\ncandidates_measured={}\npromoted={}\n",
        record.key,
        candidate.cta_m,
        candidate.cta_n,
        candidate.cta_k,
        candidate.warps_m,
        candidate.warps_n,
        candidate.num_stages,
        candidate.num_warps,
        record.baseline_hash,
        record.selected_hash,
        launch.grid_x,
        launch.grid_y,
        launch.threads,
        launch.dyn_smem_bytes,
        record.baseline_us,
        record.selected_us,
        record.candidates_measured,
        record.promoted,
    )
}

fn encode(record: &CacheRecord) -> String {
    let body = payload(record);
    format!("{body}checksum={:x}\n", Sha256::digest(body.as_bytes()))
}

fn decode(bytes: &[u8], key: &str) -> Result<CacheRecord, String> {
    let text =
        std::str::from_utf8(bytes).map_err(|_| "adaptive cache record is not UTF-8".to_string())?;
    let (body, checksum) = text
        .rsplit_once("checksum=")
        .ok_or("adaptive cache checksum is missing")?;
    let checksum = checksum
        .strip_suffix('\n')
        .ok_or("adaptive cache record is truncated")?;
    if !is_hash(checksum) || format!("{:x}", Sha256::digest(body.as_bytes())) != checksum {
        return Err("adaptive cache checksum mismatch".into());
    }
    let mut lines = body.lines();
    let mut field = |name: &str| -> Result<&str, String> {
        let line = lines
            .next()
            .ok_or_else(|| format!("adaptive cache field {name} is missing"))?;
        line.strip_prefix(name)
            .and_then(|rest| rest.strip_prefix('='))
            .ok_or_else(|| format!("unexpected adaptive cache field; expected {name}"))
    };
    macro_rules! number {
        ($name:literal) => {
            field($name)?
                .parse()
                .map_err(|_| format!("invalid adaptive cache field {}", $name))?
        };
    }
    if field("format")? != FORMAT {
        return Err("unsupported adaptive cache version".into());
    }
    let record = CacheRecord {
        key: field("key")?.to_owned(),
        candidate: AutotuneCandidate {
            cta_m: number!("cta_m"),
            cta_n: number!("cta_n"),
            cta_k: number!("cta_k"),
            warps_m: number!("warps_m"),
            warps_n: number!("warps_n"),
            num_stages: number!("num_stages"),
            num_warps: number!("num_warps"),
        },
        baseline_hash: field("baseline_hash")?.to_owned(),
        selected_hash: field("selected_hash")?.to_owned(),
        launch: LaunchConfig {
            grid_x: number!("grid_x"),
            grid_y: number!("grid_y"),
            threads: number!("threads"),
            dyn_smem_bytes: number!("dyn_smem_bytes"),
        },
        baseline_us: number!("baseline_us"),
        selected_us: number!("selected_us"),
        candidates_measured: number!("candidates_measured"),
        promoted: number!("promoted"),
    };
    if lines.next().is_some() || record.key != key {
        return Err("adaptive cache has extra fields or a mismatched key".into());
    }
    validate(&record)?;
    // Enforce one canonical representation, including whitespace and field
    // order, rather than admitting a second interpretation of signed bytes.
    if payload(&record) != body {
        return Err("adaptive cache record is not canonical".into());
    }
    Ok(record)
}

pub(super) fn load(dir: &Path, key: &str) -> Result<Option<CacheRecord>, String> {
    if !is_hash(key) {
        return Err("invalid adaptive cache lookup key".into());
    }
    let path = dir.join(format!("{key}.yjit"));
    let metadata = match fs::symlink_metadata(&path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(format!("cannot inspect adaptive cache record: {error}")),
    };
    if !metadata.is_file() || metadata.len() >= MAX_RECORD_BYTES {
        return Err("adaptive cache record is not a bounded regular file".into());
    }
    let file =
        File::open(&path).map_err(|error| format!("cannot open adaptive cache record: {error}"))?;
    if !file
        .metadata()
        .map_err(|error| format!("cannot inspect open adaptive cache record: {error}"))?
        .is_file()
    {
        return Err("adaptive cache record is not a regular file".into());
    }
    let mut bytes = Vec::new();
    file.take(MAX_RECORD_BYTES)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("cannot read adaptive cache record: {error}"))?;
    if bytes.len() as u64 >= MAX_RECORD_BYTES {
        return Err("adaptive cache record exceeds the size limit".into());
    }
    decode(&bytes, key).map(Some)
}

pub(super) fn store(dir: &Path, record: &CacheRecord, max_entries: usize) -> Result<(), String> {
    validate(record)?;
    if max_entries == 0 {
        return Err("adaptive cache capacity must be positive".into());
    }
    fs::create_dir_all(dir)
        .map_err(|error| format!("cannot create adaptive cache directory: {error}"))?;
    let bytes = encode(record);
    // Separate processes and threads must never share a temporary filename.
    // create_new also keeps a stale file from a previous process untouched.
    let (temp_path, mut file) = loop {
        let serial = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let path = dir.join(format!(
            ".{}.{}.{}.tmp",
            record.key,
            std::process::id(),
            serial
        ));
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(file) => break (path, file),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(format!("cannot create adaptive cache temporary: {error}")),
        }
    };
    let write_result = file
        .write_all(bytes.as_bytes())
        .and_then(|_| file.sync_all());
    drop(file);
    let result =
        write_result.and_then(|_| fs::rename(&temp_path, dir.join(format!("{}.yjit", record.key))));
    if let Err(error) = result {
        let cleanup = fs::remove_file(&temp_path);
        return Err(match cleanup {
            Err(cleanup) if cleanup.kind() != std::io::ErrorKind::NotFound => {
                format!("cannot write adaptive cache record: {error}; cannot clean temporary: {cleanup}")
            }
            _ => format!("cannot write adaptive cache record: {error}"),
        });
    }
    evict(dir, max_entries)
}

fn evict(dir: &Path, max_entries: usize) -> Result<(), String> {
    let mut entries = Vec::new();
    for entry in fs::read_dir(dir)
        .map_err(|error| format!("cannot list adaptive cache directory: {error}"))?
    {
        let entry =
            entry.map_err(|error| format!("cannot inspect adaptive cache entry: {error}"))?;
        let name = entry.file_name();
        let Some(key) = name.to_str().and_then(|name| name.strip_suffix(".yjit")) else {
            continue;
        };
        if !is_hash(key) {
            continue;
        }
        // Directories and links do not belong to this record store.
        let metadata = match fs::symlink_metadata(entry.path()) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => return Err(format!("cannot inspect adaptive cache entry: {error}")),
        };
        if metadata.is_file() {
            let modified = metadata
                .modified()
                .map_err(|error| format!("cannot date adaptive cache entry: {error}"))?;
            entries.push((modified, entry.path()));
        }
    }
    entries.sort_unstable();
    let excess = entries.len().saturating_sub(max_entries);
    for (_, path) in entries.into_iter().take(excess) {
        match fs::remove_file(path) {
            Ok(()) => (),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => (),
            Err(error) => return Err(format!("cannot evict adaptive cache record: {error}")),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::{Arc, Barrier};

    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> Self {
            let serial = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "y-adaptive-cache-test-{}-{serial}",
                std::process::id()
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn record(promoted: bool) -> CacheRecord {
        CacheRecord {
            key: "1".repeat(64),
            candidate: AutotuneCandidate {
                cta_m: 64,
                cta_n: 64,
                cta_k: 32,
                warps_m: 2,
                warps_n: 2,
                num_stages: 3,
                num_warps: 4,
            },
            baseline_hash: "a".repeat(64),
            selected_hash: if promoted { "b" } else { "a" }.repeat(64),
            launch: LaunchConfig {
                grid_x: 8,
                grid_y: 8,
                threads: 128,
                dyn_smem_bytes: 32768,
            },
            baseline_us: 10.0,
            selected_us: if promoted { 9.0 } else { 10.0 },
            candidates_measured: 8,
            promoted,
        }
    }

    fn write_raw(dir: &Path, record: &CacheRecord, text: impl AsRef<[u8]>) {
        fs::write(dir.join(format!("{}.yjit", record.key)), text).unwrap();
    }

    fn resign(body: &str) -> String {
        format!("{body}checksum={:x}\n", Sha256::digest(body.as_bytes()))
    }

    #[test]
    fn roundtrips_promoted_and_retained_baselines() {
        let dir = TempDir::new();
        for promoted in [false, true] {
            let record = record(promoted);
            store(&dir.0, &record, 4).unwrap();
            assert_eq!(load(&dir.0, &record.key).unwrap(), Some(record));
        }
    }

    #[test]
    fn miss_does_not_create_a_directory_and_keys_cannot_escape() {
        let dir = TempDir::new();
        let missing = dir.0.join("missing");
        assert!(load(&missing, &record(false).key).unwrap().is_none());
        assert!(!missing.exists());
        assert!(load(&dir.0, "../outside").is_err());
        let mut invalid = record(false);
        invalid.key = "A".repeat(64);
        assert!(store(&missing, &invalid, 4).is_err());
        assert!(!missing.exists());
    }

    #[test]
    fn rejects_corruption_truncation_oversize_and_wrong_key() {
        let dir = TempDir::new();
        let record = record(true);
        let valid = encode(&record);
        let corrupt = valid.replace("selected_us=9", "selected_us=8");
        for text in [
            corrupt,
            valid[..valid.len() - 1].to_owned(),
            "x".repeat(8193),
        ] {
            write_raw(&dir.0, &record, text);
            assert!(load(&dir.0, &record.key).is_err());
        }
        let other = CacheRecord {
            key: "2".repeat(64),
            ..record.clone()
        };
        write_raw(&dir.0, &record, encode(&other));
        assert!(load(&dir.0, &record.key).is_err());
    }

    #[test]
    fn rejects_resigned_nonfinite_invalid_candidates_and_geometry() {
        let dir = TempDir::new();
        let valid = record(true);
        let body = payload(&valid);
        for (from, to) in [
            ("baseline_us=10", "baseline_us=NaN"),
            ("selected_us=9", "selected_us=inf"),
            ("selected_us=9", "selected_us=0"),
            ("selected_us=9", "selected_us=11"),
            ("promoted=true", "promoted=false"),
            ("candidates_measured=8", "candidates_measured=0"),
            ("candidates_measured=8", "candidates_measured=65"),
            ("cta_m=64", "cta_m=0"),
            ("cta_m=64", "cta_m=4294967295"),
            ("warps_m=2", "warps_m=0"),
            ("warps_m=2", "warps_m=4294967295"),
            ("num_warps=4", "num_warps=32"),
            ("num_stages=3", "num_stages=4294967295"),
            ("grid_x=8", "grid_x=0"),
            ("grid_y=8", "grid_y=4294967295"),
            ("threads=128", "threads=256"),
            ("dyn_smem_bytes=32768", "dyn_smem_bytes=4294967295"),
        ] {
            write_raw(&dir.0, &valid, resign(&body.replace(from, to)));
            assert!(load(&dir.0, &valid.key).is_err(), "accepted {to}");
        }
    }

    #[test]
    fn rejects_extra_duplicate_missing_and_noncanonical_fields() {
        let dir = TempDir::new();
        let record = record(false);
        let body = payload(&record);
        for malformed in [
            format!("{body}unknown=1\n"),
            format!("{body}promoted=false\n"),
            body.replace("num_stages=3\n", ""),
            body.replace("grid_x=8", "grid_x=08"),
            body.replace(FORMAT, "y-adaptive-jit-v2"),
        ] {
            write_raw(&dir.0, &record, resign(&malformed));
            assert!(load(&dir.0, &record.key).is_err());
        }
    }

    #[test]
    fn evicts_oldest_owned_records_and_preserves_foreign_files() {
        let dir = TempDir::new();
        let foreign = ["notes.txt", "foreign.yjit", "ABC.yjit", ".temporary.tmp"];
        for name in foreign {
            fs::write(dir.0.join(name), "keep").unwrap();
        }
        fs::create_dir(dir.0.join(format!("{}.yjit", "f".repeat(64)))).unwrap();
        let first = record(false);
        store(&dir.0, &first, 2).unwrap();
        // Explicit timestamps make eviction deterministic on filesystems with
        // coarse modification-time resolution, without sleeping in tests.
        File::open(dir.0.join(format!("{}.yjit", first.key)))
            .unwrap()
            .set_times(fs::FileTimes::new().set_modified(std::time::UNIX_EPOCH))
            .unwrap();
        let second = CacheRecord {
            key: "2".repeat(64),
            ..first.clone()
        };
        let third = CacheRecord {
            key: "3".repeat(64),
            ..first.clone()
        };
        store(&dir.0, &second, 2).unwrap();
        store(&dir.0, &third, 2).unwrap();
        assert!(load(&dir.0, &first.key).unwrap().is_none());
        assert!(load(&dir.0, &second.key).unwrap().is_some());
        assert!(load(&dir.0, &third.key).unwrap().is_some());
        for name in foreign {
            assert_eq!(fs::read_to_string(dir.0.join(name)).unwrap(), "keep");
        }
        assert!(dir.0.join(format!("{}.yjit", "f".repeat(64))).is_dir());
    }

    #[test]
    fn concurrent_writers_publish_only_complete_records() {
        let dir = TempDir::new();
        let first = record(false);
        let second = record(true);
        store(&dir.0, &first, 2).unwrap();
        let barrier = Arc::new(Barrier::new(5));
        std::thread::scope(|scope| {
            for writer in 0..4 {
                let barrier = Arc::clone(&barrier);
                let path = &dir.0;
                let record = if writer % 2 == 0 { &first } else { &second };
                scope.spawn(move || {
                    barrier.wait();
                    for _ in 0..16 {
                        store(path, record, 2).unwrap();
                    }
                });
            }
            barrier.wait();
            for _ in 0..128 {
                let got = load(&dir.0, &first.key).unwrap().unwrap();
                assert!(got == first || got == second);
            }
        });
        assert_eq!(fs::read_dir(&dir.0).unwrap().count(), 1);
    }

    #[test]
    fn filesystem_errors_and_failed_rename_are_reported_and_cleaned() {
        let dir = TempDir::new();
        let record = record(false);
        let file_as_dir = dir.0.join("file");
        fs::write(&file_as_dir, "not a directory").unwrap();
        assert!(load(&file_as_dir, &record.key).is_err());
        assert!(store(&file_as_dir, &record, 2).is_err());
        let destination = dir.0.join(format!("{}.yjit", record.key));
        fs::create_dir(&destination).unwrap();
        assert!(load(&dir.0, &record.key).is_err());
        assert!(store(&dir.0, &record, 2).is_err());
        assert_eq!(fs::read_dir(&dir.0).unwrap().count(), 2);
        assert!(store(&dir.0, &record, 0).is_err());
    }
}
