//! Native runtime for the LLVM CPU JIT.
//!
//! Unlike the interpreter's C runtime, LLVM uses pointers, 64-bit lengths,
//! and byte characters. String literals are already `YStr` handles when
//! `String_new` is called; only `ystr_new` consumes a C string.
//!
//! The emitter passes both handles and addresses of handle variables for
//! references. Track our allocations to distinguish those forms without
//! mistaking string data for a runtime header. As with C, callers must supply
//! valid pointers, synchronize mutable access, and respect object lifetimes.

use std::collections::HashMap;
use std::ffi::{c_char, c_void, CStr};
use std::io::{self, Write};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use std::sync::{Mutex, MutexGuard, OnceLock};

extern "C" {
    fn malloc(size: usize) -> *mut c_void;
    fn realloc(value: *mut c_void, size: usize) -> *mut c_void;
    fn free(value: *mut c_void);
}

#[repr(C)]
struct YStr {
    data: *mut u8,
    len: i64,
    cap: i64,
}

#[repr(C)]
struct YVec {
    data: *mut u8,
    len: i64,
    cap: i64,
    elem_size: i64,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum Kind {
    String,
    Vector,
}

fn objects() -> MutexGuard<'static, HashMap<usize, Kind>> {
    static OBJECTS: OnceLock<Mutex<HashMap<usize, Kind>>> = OnceLock::new();
    match OBJECTS.get_or_init(|| Mutex::new(HashMap::new())).lock() {
        Ok(objects) => objects,
        Err(poisoned) => poisoned.into_inner(),
    }
}

struct Handle<T> {
    value: *mut T,
    slot: *mut *mut T,
}

/// `arg` must be a live runtime handle or a readable pointer-sized slot.
unsafe fn resolve<T>(arg: *const c_void, kind: Kind) -> Option<Handle<T>> {
    if arg.is_null() {
        return None;
    }
    let registry = objects();
    if registry.get(&(arg as usize)) == Some(&kind) {
        return Some(Handle {
            value: arg.cast_mut().cast(),
            slot: ptr::null_mut(),
        });
    }
    let slot = arg.cast_mut().cast::<*mut T>();
    let value = ptr::read_unaligned(slot);
    (registry.get(&(value as usize)) == Some(&kind)).then_some(Handle { value, slot })
}

unsafe extern "C" fn normalize_string_handle(arg: *const c_void) -> *mut YStr {
    guarded(ptr::null_mut(), || {
        resolve::<YStr>(arg, Kind::String).map_or(ptr::null_mut(), |handle| handle.value)
    })
}

/// Supplied directly to LLVM rather than through an overridable source name.
pub(super) fn string_handle_normalizer_address() -> usize {
    normalize_string_handle as *const () as usize
}

// Rust I/O and filesystem operations can fail without unwinding across the
// generated code's C ABI. Convert any unexpected Rust panic to a neutral
// result too; allocation failure retains Rust/libc's normal failure behavior.
fn guarded<R>(fallback: R, work: impl FnOnce() -> R) -> R {
    catch_unwind(AssertUnwindSafe(work)).unwrap_or(fallback)
}

unsafe fn string_bytes<'a>(value: *const YStr) -> &'a [u8] {
    if value.is_null() || (*value).len <= 0 || (*value).data.is_null() {
        &[]
    } else {
        std::slice::from_raw_parts((*value).data, (*value).len as usize)
    }
}

unsafe fn new_string(bytes: &[u8]) -> *mut YStr {
    let Some(cap) = bytes
        .len()
        .checked_add(1)
        .filter(|&v| v <= i64::MAX as usize)
    else {
        return ptr::null_mut();
    };
    let value = malloc(std::mem::size_of::<YStr>()).cast::<YStr>();
    if value.is_null() {
        return value;
    }
    let data = malloc(cap).cast::<u8>();
    if data.is_null() {
        free(value.cast());
        return ptr::null_mut();
    }
    ptr::copy_nonoverlapping(bytes.as_ptr(), data, bytes.len());
    *data.add(bytes.len()) = 0;
    ptr::write(
        value,
        YStr {
            data,
            len: bytes.len() as i64,
            cap: cap as i64,
        },
    );
    objects().insert(value as usize, Kind::String);
    value
}

unsafe fn string_reserve(value: *mut YStr, additional: i64) -> bool {
    let Some(needed) = (*value)
        .len
        .checked_add(additional)
        .and_then(|v| v.checked_add(1))
    else {
        return false;
    };
    if needed <= (*value).cap {
        return true;
    }
    let cap = (*value)
        .cap
        .checked_mul(2)
        .unwrap_or(needed)
        .max(8)
        .max(needed);
    let data = realloc((*value).data.cast(), cap as usize).cast::<u8>();
    if data.is_null() {
        return false;
    }
    (*value).data = data;
    (*value).cap = cap;
    true
}

unsafe extern "C" fn ystr_new(value: *const c_char) -> *mut YStr {
    guarded(ptr::null_mut(), || {
        new_string(if value.is_null() {
            &[]
        } else {
            CStr::from_ptr(value).to_bytes()
        })
    })
}

unsafe extern "C" fn ystr_clone(value: *const c_void) -> *mut YStr {
    guarded(ptr::null_mut(), || {
        let value = resolve::<YStr>(value, Kind::String).map_or(ptr::null_mut(), |h| h.value);
        new_string(string_bytes(value))
    })
}

unsafe extern "C" fn ystr_len(value: *const c_void) -> i64 {
    guarded(0, || {
        resolve::<YStr>(value, Kind::String).map_or(0, |h| (*h.value).len)
    })
}

unsafe extern "C" fn ystr_char_at(value: *const c_void, index: i64) -> u8 {
    guarded(0, || {
        let Some(value) = resolve::<YStr>(value, Kind::String) else {
            return 0;
        };
        if index < 0 || index >= (*value.value).len {
            0
        } else {
            *(*value.value).data.add(index as usize)
        }
    })
}

unsafe extern "C" fn ystr_eq(a: *const c_void, b: *const c_void) -> bool {
    guarded(false, || {
        match (
            resolve::<YStr>(a, Kind::String),
            resolve::<YStr>(b, Kind::String),
        ) {
            (Some(a), Some(b)) => string_bytes(a.value) == string_bytes(b.value),
            _ => false,
        }
    })
}

unsafe extern "C" fn ystr_eq_cstr(a: *const c_void, b: *const c_char) -> bool {
    guarded(false, || {
        let Some(a) = resolve::<YStr>(a, Kind::String) else {
            return false;
        };
        if b.is_null() {
            return false;
        }
        // LLVM wraps every source string literal in ystr_new, even when the
        // callee's name says cstr. A genuine &char/C string remains supported.
        let b_is_string = objects().get(&(b as usize)) == Some(&Kind::String);
        let b = if b_is_string {
            string_bytes(b.cast::<YStr>())
        } else {
            CStr::from_ptr(b).to_bytes()
        };
        string_bytes(a.value) == b
    })
}

unsafe extern "C" fn ystr_push(value: *const c_void, byte: u8) {
    guarded((), || {
        let Some(value) = resolve::<YStr>(value, Kind::String) else {
            return;
        };
        let value = value.value;
        if string_reserve(value, 1) {
            *(*value).data.add((*value).len as usize) = byte;
            (*value).len += 1;
            *(*value).data.add((*value).len as usize) = 0;
        }
    });
}

unsafe extern "C" fn ystr_push_str(value: *const c_void, other: *const c_void) {
    guarded((), || {
        let (Some(value), Some(other)) = (
            resolve::<YStr>(value, Kind::String),
            resolve::<YStr>(other, Kind::String),
        ) else {
            return;
        };
        let value = value.value;
        let other = other.value;
        let additional = (*other).len;
        if string_reserve(value, additional) {
            // ptr::copy also supports appending the string to itself. Resolve
            // the source data after realloc, since the header may be shared.
            ptr::copy(
                (*other).data,
                (*value).data.add((*value).len as usize),
                additional as usize,
            );
            (*value).len += additional;
            *(*value).data.add((*value).len as usize) = 0;
        }
    });
}

unsafe extern "C" fn ystr_free(value: *const c_void) {
    guarded((), || {
        let Some(value) = resolve::<YStr>(value, Kind::String) else {
            return;
        };
        objects().remove(&(value.value as usize));
        free((*value.value).data.cast());
        free(value.value.cast());
        if !value.slot.is_null() {
            ptr::write_unaligned(value.slot, ptr::null_mut());
        }
    });
}

unsafe extern "C" fn yvec_new(elem_size: i64) -> *mut YVec {
    guarded(ptr::null_mut(), || {
        if elem_size <= 0 {
            return ptr::null_mut();
        }
        let value = malloc(std::mem::size_of::<YVec>()).cast::<YVec>();
        if !value.is_null() {
            ptr::write(
                value,
                YVec {
                    data: ptr::null_mut(),
                    len: 0,
                    cap: 0,
                    elem_size,
                },
            );
            objects().insert(value as usize, Kind::Vector);
        }
        value
    })
}

unsafe extern "C" fn vec_new(elem_size: i32) -> *mut YVec {
    yvec_new(i64::from(elem_size))
}

unsafe extern "C" fn yvec_push(value: *const c_void, element: *const c_void) {
    guarded((), || {
        let Some(value) = resolve::<YVec>(value, Kind::Vector) else {
            return;
        };
        let value = value.value;
        if element.is_null() {
            return;
        }
        // A Vec element can be a borrowed pointer into this same vector.
        // Preserve its offset across realloc instead of reading freed data.
        let occupied = (*value).len.checked_mul((*value).elem_size).unwrap_or(0) as usize;
        let source_offset = (element as usize)
            .checked_sub((*value).data as usize)
            .filter(|&offset| offset < occupied);
        if (*value).len == (*value).cap {
            let Some(cap) = (*value).cap.checked_mul(2).map(|v| v.max(8)) else {
                return;
            };
            let Some(bytes) = cap.checked_mul((*value).elem_size) else {
                return;
            };
            let data = realloc((*value).data.cast(), bytes as usize).cast::<u8>();
            if data.is_null() {
                return;
            }
            (*value).data = data;
            (*value).cap = cap;
        }
        let source = source_offset.map_or(element.cast::<u8>(), |offset| (*value).data.add(offset));
        ptr::copy(
            source,
            (*value)
                .data
                .add(((*value).len * (*value).elem_size) as usize),
            (*value).elem_size as usize,
        );
        (*value).len += 1;
    });
}

unsafe extern "C" fn yvec_get(value: *const c_void, index: i64) -> *mut c_void {
    guarded(ptr::null_mut(), || {
        let Some(value) = resolve::<YVec>(value, Kind::Vector) else {
            return ptr::null_mut();
        };
        let value = value.value;
        if index < 0 || index >= (*value).len {
            ptr::null_mut()
        } else {
            (*value)
                .data
                .add((index * (*value).elem_size) as usize)
                .cast()
        }
    })
}

unsafe extern "C" fn yvec_get_char(value: *const c_void, index: i64) -> u8 {
    let element = yvec_get(value, index).cast::<u8>();
    if element.is_null() {
        0
    } else {
        *element
    }
}

unsafe extern "C" fn yvec_len(value: *const c_void) -> i64 {
    guarded(0, || {
        resolve::<YVec>(value, Kind::Vector).map_or(0, |h| (*h.value).len)
    })
}

unsafe extern "C" fn yvec_free(value: *const c_void) {
    guarded((), || {
        let Some(value) = resolve::<YVec>(value, Kind::Vector) else {
            return;
        };
        objects().remove(&(value.value as usize));
        free((*value.value).data.cast());
        free(value.value.cast());
        if !value.slot.is_null() {
            ptr::write_unaligned(value.slot, ptr::null_mut());
        }
    });
}

unsafe fn output(value: *const c_void, newline: bool) {
    let value = resolve::<YStr>(value, Kind::String).map_or(ptr::null_mut(), |h| h.value);
    let mut stdout = io::stdout().lock();
    let _ = stdout.write_all(string_bytes(value));
    if newline {
        let _ = stdout.write_all(b"\n");
    }
    let _ = stdout.flush();
}

unsafe extern "C" fn print(value: *const c_void) {
    guarded((), || output(value, false));
}

unsafe extern "C" fn println(value: *const c_void) {
    guarded((), || output(value, true));
}

extern "C" fn print_int(value: i64) {
    guarded((), || {
        let mut stdout = io::stdout().lock();
        let _ = write!(stdout, "{value}");
        let _ = stdout.flush();
    });
}

extern "C" fn yprint_char(value: u8) {
    guarded((), || {
        let mut stdout = io::stdout().lock();
        let _ = stdout.write_all(&[value]);
        let _ = stdout.flush();
    });
}

#[cfg(unix)]
unsafe extern "C" fn yfile_read_to_string(path: *const c_void) -> *mut YStr {
    use std::os::unix::ffi::OsStrExt;
    guarded(ptr::null_mut(), || {
        let path = resolve::<YStr>(path, Kind::String).map_or(ptr::null_mut(), |h| h.value);
        let content =
            std::fs::read(std::ffi::OsStr::from_bytes(string_bytes(path))).unwrap_or_default();
        new_string(&content)
    })
}

#[cfg(unix)]
unsafe extern "C" fn yfile_write(path: *const c_void, content: *const c_void) {
    use std::os::unix::ffi::OsStrExt;
    guarded((), || {
        let (Some(path), Some(content)) = (
            resolve::<YStr>(path, Kind::String),
            resolve::<YStr>(content, Kind::String),
        ) else {
            return;
        };
        let _ = std::fs::write(
            std::ffi::OsStr::from_bytes(string_bytes(path.value)),
            string_bytes(content.value),
        );
    });
}

unsafe extern "C" fn str_to_i64(value: *const c_void) -> i64 {
    guarded(0, || {
        let Some(value) = resolve::<YStr>(value, Kind::String) else {
            return 0;
        };
        std::str::from_utf8(string_bytes(value.value))
            .ok()
            .and_then(|text| text.trim().parse().ok())
            .unwrap_or(0)
    })
}

extern "C" fn ychar_to_ascii(value: u8) -> i32 {
    i32::from(value)
}

extern "C" fn math_sqrt(value: f32) -> f32 {
    value.sqrt()
}

extern "C" fn math_fmin(a: f32, b: f32) -> f32 {
    a.min(b)
}

extern "C" fn math_fmax(a: f32, b: f32) -> f32 {
    a.max(b)
}

pub(super) fn symbols() -> Vec<(&'static str, usize)> {
    let mut symbols = vec![
        ("print", print as *const () as usize),
        ("println", println as *const () as usize),
        ("print_int", print_int as *const () as usize),
        ("yprint_str", print as *const () as usize),
        ("yprintln_str", println as *const () as usize),
        ("yprint_int", print_int as *const () as usize),
        ("yprint_char", yprint_char as *const () as usize),
        ("String_new", ystr_clone as *const () as usize),
        ("ystr_new", ystr_new as *const () as usize),
        ("String_clone", ystr_clone as *const () as usize),
        ("ystr_clone", ystr_clone as *const () as usize),
        ("String_len", ystr_len as *const () as usize),
        ("ystr_len", ystr_len as *const () as usize),
        ("String_char_at", ystr_char_at as *const () as usize),
        ("ystr_char_at", ystr_char_at as *const () as usize),
        ("String_eq", ystr_eq as *const () as usize),
        ("ystr_eq", ystr_eq as *const () as usize),
        ("String_eq_cstr", ystr_eq_cstr as *const () as usize),
        ("ystr_eq_cstr", ystr_eq_cstr as *const () as usize),
        ("String_push", ystr_push as *const () as usize),
        ("ystr_push", ystr_push as *const () as usize),
        ("String_push_str", ystr_push_str as *const () as usize),
        ("ystr_push_str", ystr_push_str as *const () as usize),
        ("String_free", ystr_free as *const () as usize),
        ("ystr_free", ystr_free as *const () as usize),
        ("Vec_new", vec_new as *const () as usize),
        ("yvec_new", yvec_new as *const () as usize),
        ("Vec_push", yvec_push as *const () as usize),
        ("yvec_push", yvec_push as *const () as usize),
        ("Vec_get", yvec_get as *const () as usize),
        ("yvec_get", yvec_get as *const () as usize),
        ("Vec_get_char", yvec_get_char as *const () as usize),
        ("yvec_get_char", yvec_get_char as *const () as usize),
        ("Vec_len", yvec_len as *const () as usize),
        ("yvec_len", yvec_len as *const () as usize),
        ("Vec_free", yvec_free as *const () as usize),
        ("yvec_free", yvec_free as *const () as usize),
        ("str_to_i64", str_to_i64 as *const () as usize),
        ("ychar_to_ascii", ychar_to_ascii as *const () as usize),
        ("math_sqrt", math_sqrt as *const () as usize),
        ("math_fmin", math_fmin as *const () as usize),
        ("math_fmax", math_fmax as *const () as usize),
    ];
    #[cfg(unix)]
    symbols.extend([
        (
            "File_read_to_string",
            yfile_read_to_string as *const () as usize,
        ),
        (
            "yfile_read_to_string",
            yfile_read_to_string as *const () as usize,
        ),
        ("File_write", yfile_write as *const () as usize),
        ("yfile_write", yfile_write as *const () as usize),
    ]);
    symbols
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn headers_match_compiler_visible_cpu_jit_layouts() {
        // LlvmEmitter's closed-local queries use exactly these pointer/i64
        // fields. The interpreter/AOT runtime has a different handle ABI.
        assert_eq!(std::mem::align_of::<YStr>(), 8);
        assert_eq!(std::mem::size_of::<YStr>(), 24);
        assert_eq!(std::mem::offset_of!(YStr, data), 0);
        assert_eq!(std::mem::offset_of!(YStr, len), 8);
        assert_eq!(std::mem::offset_of!(YStr, cap), 16);
        assert_eq!(std::mem::align_of::<YVec>(), 8);
        assert_eq!(std::mem::size_of::<YVec>(), 32);
        assert_eq!(std::mem::offset_of!(YVec, data), 0);
        assert_eq!(std::mem::offset_of!(YVec, len), 8);
        assert_eq!(std::mem::offset_of!(YVec, cap), 16);
        assert_eq!(std::mem::offset_of!(YVec, elem_size), 24);
    }

    #[test]
    fn strings_support_handle_references_binary_contents_and_self_append() {
        unsafe {
            let mut value = ystr_new(c"ab".as_ptr());
            let reference = (&mut value as *mut *mut YStr).cast::<c_void>();
            assert_eq!(ystr_len(reference), 2);
            ystr_push(reference, 0);
            ystr_push(value.cast(), b'c');
            let cloned = ystr_clone(reference);
            assert!(ystr_eq(value.cast(), cloned.cast()));
            assert_eq!(string_bytes(cloned), b"ab\0c");
            ystr_push_str(reference, reference);
            assert_eq!(string_bytes(value), b"ab\0cab\0c");
            assert_eq!(ystr_char_at(reference, -1), 0);
            assert_eq!(ystr_char_at(reference, 7), b'c');
            assert_eq!(ystr_char_at(reference, 8), 0);
            ystr_free(cloned.cast());
            ystr_free(reference);
            assert!(value.is_null());
            ystr_free(reference);
        }
    }

    #[test]
    fn equality_accepts_c_text_and_llvm_wrapped_literals() {
        unsafe {
            let value = ystr_new(c"hello".as_ptr());
            let other = ystr_new(c"hello".as_ptr());
            assert!(ystr_eq_cstr(value.cast(), c"hello".as_ptr()));
            assert!(ystr_eq_cstr(value.cast(), other.cast()));
            assert!(!ystr_eq_cstr(value.cast(), c"hell".as_ptr()));
            ystr_free(value.cast());
            ystr_free(other.cast());
        }
    }

    #[test]
    fn vectors_preserve_typed_values_and_sources_across_growth() {
        unsafe {
            assert!(yvec_new(0).is_null());
            assert!(vec_new(-1).is_null());
            let mut value = vec_new(std::mem::size_of::<i64>() as i32);
            let reference = (&mut value as *mut *mut YVec).cast::<c_void>();
            for element in 0_i64..8 {
                yvec_push(reference, (&element as *const i64).cast());
            }
            // This forces realloc while the source is inside the old buffer.
            yvec_push(reference, yvec_get(value.cast(), 3));
            assert_eq!(yvec_len(reference), 9);
            assert_eq!(*yvec_get(reference, 8).cast::<i64>(), 3);
            assert!(yvec_get(reference, -1).is_null());
            assert!(yvec_get(reference, 9).is_null());
            yvec_free(reference);
            assert!(value.is_null());
        }
    }

    #[cfg(unix)]
    #[test]
    fn file_callbacks_round_trip_binary_strings_and_missing_files() {
        unsafe {
            let path = std::env::temp_dir().join(format!(
                "y-jit-runtime-{}-{:?}",
                std::process::id(),
                std::thread::current().id()
            ));
            use std::os::unix::ffi::OsStrExt;
            let path_value = new_string(path.as_os_str().as_bytes());
            let content = new_string(b"a\0b\nc");
            yfile_write(path_value.cast(), content.cast());
            let read = yfile_read_to_string(path_value.cast());
            assert!(ystr_eq(content.cast(), read.cast()));
            std::fs::remove_file(path).unwrap();
            let missing = yfile_read_to_string(path_value.cast());
            assert_eq!(ystr_len(missing.cast()), 0);
            for value in [path_value, content, read, missing] {
                ystr_free(value.cast());
            }
        }
    }
}
