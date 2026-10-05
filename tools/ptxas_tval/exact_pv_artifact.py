"""Bind exact_pv's existing translation result to the cubin loaded by CUDA.

Build in a private directory, validate the PTX against the disassembly of that
cubin, then publish the three artifacts and a SHA-256 receipt together. Loading
never invokes a compiler: it checks the receipt and loads the checked ELF bytes.
The receipt records a trusted local validation result, not a signed certificate
or a proof of the validator, ISA, caller preconditions, or CUDA loader.
"""
import argparse
from contextlib import contextmanager
import ctypes
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile
if __package__:
    from .exact_pv_subject import require_proved_subject
else:
    from exact_pv_subject import require_proved_subject


class ArtifactError(RuntimeError):
    pass


_FIXED = {
    "format": "y-exact-pv-cubin-v1",
    "entry": "exact_pv",
    "target": "sm_89",
    "optimization": "1",
    "validator": "loopval",
    "verdict": "VALIDATED",
}
_FILES = {"ptx_sha256": "exact_pv.ptx", "sass_sha256": "exact_pv.sass",
          "cubin_sha256": "exact_pv.cubin"}


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _read(path):
    try:
        return path.read_bytes()
    except OSError as e:
        raise ArtifactError(f"cannot read validated artifact {path}: {e}") from e


def _receipt(data):
    try:
        fields = {}
        for line in data.decode("utf-8").splitlines():
            key, value = line.split("=", 1)
            if key in fields:
                raise ValueError(f"duplicate field {key}")
            fields[key] = value
        if fields.keys() != _FIXED.keys() | _FILES.keys() | {"obligations"}:
            raise ValueError("missing or unknown receipt fields")
        if any(fields[k] != v for k, v in _FIXED.items()):
            raise ValueError("unrecognized format, entry, target, or validation result")
        if not re.fullmatch(r"[1-9][0-9]*", fields["obligations"]):
            raise ValueError("no successful validation obligations")
        for key in _FILES:
            if not re.fullmatch(r"[0-9a-f]{64}", fields[key]):
                raise ValueError(f"invalid SHA-256 identity: {key}")
        return fields
    except (UnicodeError, ValueError) as e:
        raise ArtifactError(f"invalid validation receipt: {e}") from e


def _check_cubin(data):
    """Recognize the exact supported CUDA ELF format, target, and entry.

    This is an identity/target check, not an instruction decoder. Format v1
    accepts CUDA ELF ABI 8, sm_89 only (the format emitted by the toolchain
    used by this validator). Unknown encodings fail instead of guessing the
    target from an ELF flags layout belonging to another CUDA ABI.
    """
    def require(ok, why):
        if not ok:
            raise ArtifactError(f"unidentified or incompatible cubin: {why}")

    def span(offset, size):
        require(offset <= len(data) and size <= len(data) - offset, "truncated ELF")
        return data[offset:offset + size]

    require(len(data) >= 64, "missing ELF header")
    require(data[:9] == b"\x7fELF\x02\x01\x01\x41\x08", "expected CUDA ELF64 ABI 8")
    h = struct.unpack_from("<HHIQQQIHHHHHH", data, 16)
    kind, machine, version, _, phoff, shoff, flags, ehsize, phsize, phnum, shsize, shnum, names_index = h
    require((kind, machine, version, ehsize) == (2, 190, 1, 64), "expected CUDA ET_EXEC")
    require(flags == 0x06005904, "expected sm_89 flags 0x06005904")
    if phnum:
        require(phsize == 56, "unsupported program headers")
        span(phoff, phsize * phnum)
    require(shnum > 0 and shsize == 64 and 0 < names_index < shnum, "invalid section table")
    span(shoff, shsize * shnum)
    sections = [struct.unpack_from("<IIQQQQIIQQ", data, shoff + i * shsize)
                for i in range(shnum)]
    names_header = sections[names_index]
    require(names_header[1] == 3, "section names are not a string table")
    names = span(names_header[4], names_header[5])
    text = []
    for section in sections:
        at = section[0]
        require(at < len(names), "invalid section name offset")
        end = names.find(b"\0", at)
        require(end >= 0, "unterminated section name")
        name = names[at:end]
        if name.startswith(b".text."):
            text.append(name)
            require(section[1] == 1 and section[2] & 6 == 6 and section[5] > 0,
                    "invalid executable section")
            span(section[4], section[5])
    require(text == [b".text.exact_pv"], "expected exactly the exact_pv entry")


@dataclass(frozen=True)
class ValidatedExactPv:
    _directory: Path
    _receipt_bytes: bytes

    @property
    def cubin_sha256(self):
        return _receipt(self._receipt_bytes)["cubin_sha256"]

    def checked_image(self):
        """Recheck pinned identity and return the SAME bytes the loader uses."""
        receipt = _read(self._directory / "receipt.txt")
        if receipt != self._receipt_bytes:
            raise ArtifactError("validation receipt changed after binding")
        fields = _receipt(receipt)
        artifacts = {}
        for key, name in _FILES.items():
            data = _read(self._directory / name)
            if _sha(data) != fields[key]:
                raise ArtifactError(f"SHA-256 mismatch: validated artifact {name} changed")
            artifacts[name] = data
        try:
            require_proved_subject(artifacts['exact_pv.ptx'])
        except ValueError as error:
            raise ArtifactError(str(error)) from error
        image = artifacts["exact_pv.cubin"]
        _check_cubin(image)
        return image

    def load(self, cuda):
        """Load the checked cubin into the current sm_89 context, with no JIT."""
        def check(result, operation):
            if result != 0:
                raise ArtifactError(f"{operation} failed (CUresult {result})")

        device = ctypes.c_int()
        check(cuda.cuCtxGetDevice(ctypes.byref(device)), "cuCtxGetDevice")
        major, minor = ctypes.c_int(), ctypes.c_int()
        check(cuda.cuDeviceGetAttribute(ctypes.byref(major), 75, device), "compute capability major")
        check(cuda.cuDeviceGetAttribute(ctypes.byref(minor), 76, device), "compute capability minor")
        if (major.value, minor.value) != (8, 9):
            raise ArtifactError(f"validated exact_pv requires sm_89; device is sm_{major.value}{minor.value}")
        # Query the device before the final read/check. Never pass a pathname to
        # CUDA: reopening it after hashing would reintroduce the artifact gap.
        image = self.checked_image()
        buffer = (ctypes.c_ubyte * len(image)).from_buffer_copy(image)
        module = ctypes.c_void_p()
        check(cuda.cuModuleLoadData(ctypes.byref(module), buffer), "load validated cubin")
        if not module.value:
            raise ArtifactError("CUDA did not identify the loaded cubin module")
        return module


def open_verified(bundle_dir, expected_ptx_sha256=None):
    directory = Path(bundle_dir).resolve()
    receipt = _read(directory / "receipt.txt")
    fields = _receipt(receipt)
    if expected_ptx_sha256 is not None and fields["ptx_sha256"] != expected_ptx_sha256:
        raise ArtifactError("validation receipt identifies different PTX")
    artifact = ValidatedExactPv(directory, receipt)
    artifact.checked_image()
    return artifact


def _run(command, timeout=60):
    try:
        return subprocess.run(command, capture_output=True, check=True, timeout=timeout).stdout
    except (OSError, subprocess.SubprocessError) as e:
        detail = getattr(e, "stderr", b"") or b""
        raise ArtifactError(f"{command[0]} failed: {e}\n{detail.decode(errors='replace')}") from e


_VALIDATION_RESULT_FORMAT = "y-exact-pv-loopval-result-v1"
_VALIDATION_CHILD = """import json, sys
sys.path.insert(0, sys.argv[1])
import loopval
verdict, detail, obligations = loopval.validate(
    sys.argv[2], sys.argv[3], budget=int(sys.argv[4]), mode='wide', verbose=False)
print(json.dumps({'format': 'y-exact-pv-loopval-result-v1', 'verdict': verdict,
                  'detail': detail, 'obligations': obligations}))
"""


def _validate_translation(ptx, sass, budget):
    """Run the existing loop validator with a fresh Z3 construction history.

    Multiplier operand canonicalization depends on Z3 node IDs. Unrelated
    validation in this interpreter can therefore change an inline verdict;
    frontier.structural already isolates validation for this reason. Use the
    caller's interpreter and keep the same per-obligation solver budget. The
    separate process has a finite wall limit of max(60, 32 * budget) seconds.
    """
    command = [sys.executable, "-c", _VALIDATION_CHILD,
               str(Path(__file__).resolve().parent), str(ptx), str(sass), str(budget)]
    output = _run(command, timeout=max(60, 32 * budget))
    def fields(pairs):
        result = dict(pairs)
        if len(result) != len(pairs):
            raise ValueError("duplicate subprocess result field")
        return result

    try:
        result = json.loads(output.decode("utf-8"), object_pairs_hook=fields)
    except (UnicodeError, ValueError) as e:
        raise ArtifactError(f"invalid translation-validation subprocess result: {e}") from e
    if (type(result) is not dict
            or result.keys() != {"format", "verdict", "detail", "obligations"}
            or result["format"] != _VALIDATION_RESULT_FORMAT
            or type(result["verdict"]) is not str
            or result["verdict"] not in {"VALIDATED", "UNPROVED", "REFUSED"}
            or type(result["detail"]) is not str
            or type(result["obligations"]) is not int
            or not 0 <= result["obligations"] <= (1 << 64) - 1):
        raise ArtifactError("invalid translation-validation subprocess result schema")
    return result["verdict"], result["detail"], result["obligations"]


@contextmanager
def _reserve_destination(target):
    """Claim a new name before validation without publishing a receipt.

    mkdir is exclusive, unlike POSIX rename onto an existing empty directory.
    A competing builder cannot claim this destination during validation.
    Cleanup removes only an empty reservation, preserving any foreign content.
    """
    try:
        target.mkdir()
    except FileExistsError as e:
        raise ArtifactError(f"artifact destination already exists: {target}") from e
    try:
        yield
    finally:
        try:
            target.rmdir()
        except OSError:
            # A published bundle or a directory populated by another process
            # must survive cleanup. Neither is an empty reservation anymore.
            pass


def build(ptx_path, bundle_dir, budget=60):
    """Publish a new bundle only after the existing loop validator accepts it.

    Existing directories are refused, so a failed rebuild cannot be mistaken
    for a fresh acceptance of an old receipt. Consumers receive no artifact on
    failure. Input and tool outputs are checked for changes across validation.
    """
    if type(budget) is not int or budget <= 0:
        raise ArtifactError("validation budget must be a positive integer")
    target = Path(bundle_dir).resolve()
    if target.exists():
        raise ArtifactError(f"artifact destination already exists: {target}")
    source = _read(Path(ptx_path))
    try:
        require_proved_subject(source)
    except ValueError as error:
        raise ArtifactError(str(error)) from error
    try:
        text = source.decode("utf-8")
    except UnicodeError as e:
        raise ArtifactError("PTX input is not UTF-8") from e
    # These imports are intentionally lazy: loading a retained bundle requires
    # neither Z3 nor a compiler. The build still uses the original validator.
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    try:
        import loopcfg
        import ptxsource
    except ImportError as e:
        raise ArtifactError(f"translation validator unavailable: {e}") from e
    try:
        subject = ptxsource.strip_comments(text)
    except Exception as e:
        raise ArtifactError(f"cannot identify PTX input: {e}") from e
    targets = re.findall(r"(?m)^[ \t]*\.target[ \t]+([^\r\n]+)", subject)
    if [target.strip() for target in targets] != ["sm_89"]:
        raise ArtifactError("verified exact_pv requires an unqualified .target sm_89")
    target.parent.mkdir(parents=True, exist_ok=True)
    with _reserve_destination(target), tempfile.TemporaryDirectory(
            prefix=".exact_pv_validation_", dir=target.parent) as temporary:
        stage = Path(temporary)
        ptx, cubin, sass = (stage / f"exact_pv.{ext}" for ext in ("ptx", "cubin", "sass"))
        ptx.write_bytes(source)
        if loopcfg.ptx_entry_points(str(ptx)) != ["exact_pv"]:
            raise ArtifactError("expected exactly one PTX entry: exact_pv")
        _run(["ptxas", "-O1", "-arch=sm_89", str(ptx), "-o", str(cubin)])
        original_cubin = _read(cubin)
        _check_cubin(original_cubin)
        sass.write_bytes(_run(["nvdisasm", "-c", str(cubin)]))
        original_sass = _read(sass)
        # The disassembler identifies the target independently of our ELF check.
        if re.findall(rb"(?m)^\s*\.target\s+([^\r\n]+)", original_sass) != [b"sm_89"]:
            raise ArtifactError("disassembly target is not sm_89")
        try:
            verdict, detail, obligations = _validate_translation(str(ptx), str(sass), budget=budget)
        except Exception as e:
            raise ArtifactError(f"translation validation refused: {e}") from e
        if verdict != "VALIDATED" or type(obligations) is not int or obligations <= 0:
            raise ArtifactError(f"translation validation did not accept exact_pv: {verdict}: {detail}")
        originals = {"exact_pv.ptx": source, "exact_pv.cubin": original_cubin,
                     "exact_pv.sass": original_sass}
        for name, data in originals.items():
            if _read(stage / name) != data:
                raise ArtifactError(f"artifact changed during validation: {name}")
        fields = dict(_FIXED, obligations=str(obligations))
        fields.update({key: _sha(originals[name]) for key, name in _FILES.items()})
        (stage / "receipt.txt").write_text("".join(f"{key}={value}\n" for key, value in fields.items()))
        open_verified(stage, _sha(source))
        # Replace our empty reservation; rename refuses if another process
        # populated it. The receipt and artifacts appear together.
        try:
            stage.rename(target)
        except OSError as e:
            raise ArtifactError(f"cannot publish validated artifact at {target}: {e}") from e
    return open_verified(target, _sha(source))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("build", help="assemble, validate, and retain a new exact_pv bundle")
    create.add_argument("ptx")
    create.add_argument("bundle")
    create.add_argument("--budget", type=int, default=60)
    inspect = commands.add_parser("check", help="check the retained receipt and artifact identities")
    inspect.add_argument("bundle")
    args = parser.parse_args()
    try:
        artifact = (build(args.ptx, args.bundle, args.budget) if args.command == "build"
                    else open_verified(args.bundle))
        print(f"BOUND exact_pv sm_89 cubin SHA-256 {artifact.cubin_sha256}")
        return 0
    except (ArtifactError, OSError) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
