"""PTX cache regressions using isolated files and a fake compiler; no GPU needed."""

import ctypes
from contextlib import ExitStack
import hashlib
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from y_lang import compiler


class FakeNativeLibrary:
    def __init__(self, identity):
        self.identity = identity
        self.buffers = []
        self.y_compile_to_ptx = mock.Mock(side_effect=self.compile)
        self.y_free_string = mock.Mock()
        self.y_autotune_search_space_json = mock.Mock()
        self.y_autotune_select_config_json = mock.Mock()

    def compile(self, source, target, error_out):
        ptx = b"PTX:" + self.identity + b":" + target + b":" + source
        buffer = ctypes.create_string_buffer(ptx)
        self.buffers.append(buffer)
        return ctypes.addressof(buffer)


class TestPtxCache(unittest.TestCase):
    SOURCE = "kernel cache_test() {}"

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="y-ptx-cache-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.library = self.root / "liby.so"
        self.library.write_bytes(b"compiler-v1")
        self.profile = self.root / ".ysu_hw_profile"
        self.profile.write_text("SM_VERSION=8.0\nSM_COUNT=80\n", encoding="utf-8")
        self.cache = self.root / "cache"
        self.libraries = []

        instance = compiler.YCompilerLib._instance
        jit_cache = compiler._JIT_CACHE.copy()
        stats = compiler._CACHE_STATS.copy()
        self.addCleanup(self.restore_state, instance, jit_cache, stats)
        compiler.YCompilerLib._instance = None
        compiler._JIT_CACHE.clear()
        compiler._CACHE_STATS.update(mem_hits=0, disk_hits=0, misses=0)

        contexts = ExitStack()
        self.addCleanup(contexts.close)
        contexts.enter_context(mock.patch.dict(os.environ, {
            "Y_LIB_PATH": str(self.library), "YSU_CACHE_DIR": str(self.cache),
        }))
        contexts.enter_context(mock.patch.object(Path, "cwd", return_value=self.root))
        contexts.enter_context(mock.patch.object(compiler.ctypes, "CDLL", side_effect=self.load))
        self.verify_patch = mock.patch.object(compiler, "_verify_loaded_library")
        self.verify_patch.start()
        self.addCleanup(self.verify_patch.stop)

    @staticmethod
    def restore_state(instance, jit_cache, stats):
        compiler.YCompilerLib._instance = instance
        compiler._JIT_CACHE.clear()
        compiler._JIT_CACHE.update(jit_cache)
        compiler._CACHE_STATS.clear()
        compiler._CACHE_STATS.update(stats)

    def load(self, path):
        native = FakeNativeLibrary(Path(path).read_bytes())
        self.libraries.append(native)
        return native

    def compile(self, target="sm_80"):
        return compiler.compile_to_ptx(self.SOURCE, target)

    def rebuild(self, identity=b"compiler-v2"):
        # Cargo replaces the artifact rather than mutating the loaded inode.
        replacement = self.root / "replacement.so"
        replacement.write_bytes(identity)
        replacement.replace(self.library)

    def test_unchanged_compiler_hits_memory_and_disk(self):
        expected = self.compile()
        self.assertEqual(self.compile(), expected)
        self.assertEqual(compiler.get_cache_stats()["mem_hits"], 1)
        compiler._JIT_CACHE.clear()
        self.assertEqual(self.compile(), expected)
        stats = compiler.get_cache_stats()
        self.assertEqual(stats["disk_hits"], 1)
        self.assertEqual(stats["misses"], 1)
        self.assertEqual(stats["cached_files_count"], 1)
        self.assertEqual(self.libraries[0].y_compile_to_ptx.call_count, 1)

    def test_warm_hit_does_not_hash_or_reload_library(self):
        expected = self.compile()
        original_open = Path.open

        def guarded_open(path, *args, **kwargs):
            self.assertNotEqual(path, self.library, "warm hit re-read the compiler binary")
            return original_open(path, *args, **kwargs)

        with mock.patch.object(Path, "open", guarded_open):
            self.assertEqual(self.compile(), expected)
        self.assertEqual(len(self.libraries), 1)

    def test_unchanged_compiler_reuses_disk_cache_in_new_instance(self):
        expected = self.compile()
        compiler.YCompilerLib._instance = None
        compiler._JIT_CACHE.clear()
        self.assertEqual(self.compile(), expected)
        self.assertEqual(len(self.libraries), 2)
        self.assertEqual(self.libraries[1].y_compile_to_ptx.call_count, 0)
        self.assertEqual(compiler.get_cache_stats()["disk_hits"], 1)

    def test_compiler_change_invalidates_existing_memory_and_disk_keys(self):
        old = self.compile()
        self.rebuild()
        # Simulate a fresh process, while keeping memory to prove that the
        # compiler component of the key also excludes an old in-memory entry.
        compiler.YCompilerLib._instance = None
        new = self.compile()
        self.assertNotEqual(old, new)
        self.assertIn("compiler-v2", new)
        stats = compiler.get_cache_stats()
        self.assertEqual(stats["mem_hits"], 0)
        self.assertEqual(stats["disk_hits"], 0)
        self.assertEqual(stats["misses"], 2)
        self.assertEqual(stats["cached_files_count"], 2)
        compiler._JIT_CACHE.clear()
        self.assertEqual(self.compile(), new)
        self.assertEqual(compiler.get_cache_stats()["disk_hits"], 1)

    def test_rebuild_in_loaded_process_refuses_stale_memory_hit(self):
        self.compile()
        self.rebuild()
        with self.assertRaisesRegex(RuntimeError, "Restart Python"):
            self.compile()
        self.assertEqual(compiler._JIT_CACHE, {})
        self.assertEqual(self.libraries[0].y_compile_to_ptx.call_count, 1)
        self.assertEqual(compiler.get_cache_stats()["mem_hits"], 0)

    def test_rebuild_in_loaded_process_refuses_stale_disk_hit(self):
        self.compile()
        compiler._JIT_CACHE.clear()
        self.rebuild()
        with self.assertRaisesRegex(RuntimeError, "Restart Python"):
            self.compile()
        self.assertEqual(compiler.get_cache_stats()["disk_hits"], 0)

    def test_in_place_library_update_requires_restart(self):
        self.compile()
        self.library.write_bytes(b"compiler-v2-with-new-contents")
        with self.assertRaisesRegex(RuntimeError, "Restart Python"):
            self.compile()
        self.assertEqual(compiler.get_cache_stats()["mem_hits"], 0)

    def test_rebuild_during_initial_load_is_rejected(self):
        def rebuild_before_load(path):
            self.rebuild()
            return self.load(path)

        with mock.patch.object(compiler.ctypes, "CDLL", side_effect=rebuild_before_load):
            with self.assertRaisesRegex(RuntimeError, "changed while loading"):
                self.compile()
        self.assertIsNone(compiler.YCompilerLib._instance)
        self.assertEqual(compiler.get_cache_stats()["cached_files_count"], 0)

    def test_changed_y_lib_path_requires_restart_then_uses_selected_compiler(self):
        old = self.compile()
        other = self.root / "other-liby.so"
        other.write_bytes(b"selected-compiler")
        os.environ["Y_LIB_PATH"] = str(other)
        with self.assertRaisesRegex(RuntimeError, "Y_LIB_PATH"):
            self.compile()
        compiler.YCompilerLib._instance = None
        self.assertNotEqual(self.compile(), old)
        self.assertEqual(compiler.YCompilerLib.get_instance().lib_path, str(other))

    def test_target_architectures_have_separate_entries(self):
        sm80 = self.compile("sm_80")
        sm90 = self.compile("sm_90a")
        self.assertNotEqual(sm80, sm90)
        self.assertEqual(self.compile("sm_80"), sm80)
        self.assertEqual(self.compile("sm_90a"), sm90)
        self.assertEqual(compiler.get_cache_stats()["misses"], 2)
        self.assertEqual(compiler.get_cache_stats()["mem_hits"], 2)

    def test_legacy_cache_file_is_ignored(self):
        self.cache.mkdir()
        legacy = hashlib.sha256(("sm_80:" + self.SOURCE).encode()).hexdigest()
        (self.cache / (legacy + ".ptx")).write_text("stale legacy PTX", encoding="utf-8")
        self.assertNotEqual(self.compile(), "stale legacy PTX")
        self.assertEqual(compiler.get_cache_stats()["disk_hits"], 0)

    def test_profile_change_invalidates_auto_target_memory_and_disk(self):
        self.compile("auto")
        self.profile.write_text("SM_VERSION=9.0\nSM_COUNT=120\n", encoding="utf-8")
        self.compile("auto")
        self.assertEqual(self.libraries[0].y_compile_to_ptx.call_count, 2)
        self.assertEqual(compiler.get_cache_stats()["misses"], 2)
        self.assertEqual(compiler.get_cache_stats()["cached_files_count"], 2)

    def test_explicit_target_still_depends_on_profile_tuning(self):
        self.compile()
        self.profile.write_text("SM_VERSION=8.0\nSM_COUNT=120\n", encoding="utf-8")
        self.compile()
        self.assertEqual(self.libraries[0].y_compile_to_ptx.call_count, 2)

    def test_missing_profile_bypasses_auto_cache(self):
        self.profile.unlink()
        self.compile("auto")
        self.compile("auto")
        self.assertEqual(self.libraries[0].y_compile_to_ptx.call_count, 2)
        self.assertEqual(compiler._JIT_CACHE, {})
        self.assertEqual(compiler.get_cache_stats()["cached_files_count"], 0)

    def test_profile_created_during_compile_is_not_mislabeled(self):
        self.profile.unlink()

        def compile_and_probe(source, target, error_out):
            self.profile.write_text("SM_VERSION=8.9\n", encoding="utf-8")
            return self.libraries[0].compile(source, target, error_out)

        lib = compiler.YCompilerLib.get_instance()
        lib.lib.y_compile_to_ptx.side_effect = compile_and_probe
        self.compile("auto")
        self.assertEqual(compiler.get_cache_stats()["cached_files_count"], 0)
        self.compile("auto")
        self.compile("auto")
        self.assertEqual(compiler.get_cache_stats()["misses"], 2)
        self.assertEqual(compiler.get_cache_stats()["mem_hits"], 1)

    def test_codegen_environment_change_invalidates_cache(self):
        with mock.patch.dict(os.environ, {"Y_CTA_OVERRIDE": "64,64,32"}):
            self.compile()
        with mock.patch.dict(os.environ, {"Y_CTA_OVERRIDE": "128,64,32"}):
            self.compile()
        self.assertEqual(self.libraries[0].y_compile_to_ptx.call_count, 2)

    def verify_mapping(self, inode_delta=0, path=None, suffix=""):
        """Run the real check against one synthesized `/proc/self/maps` line.

        The line's device is deliberately NOT the one `stat()` reports: btrfs
        and overlayfs show the superblock's device there, so a check that
        compared devices refused every real library on those filesystems.
        """
        self.verify_patch.stop()
        self.addCleanup(self.verify_patch.start)
        buffer = ctypes.create_string_buffer(b"function")
        address = ctypes.addressof(buffer)
        lib = SimpleNamespace(y_compile_to_ptx=ctypes.c_void_p(address))
        signature = compiler._library_signature(self.library)
        device = "{:x}:{:x}".format(os.major(signature[0]) + 7, os.minor(signature[0]) + 3)
        mapped = path if path is not None else os.path.realpath(self.library)
        mapping = "{:x}-{:x} r-xp 00000000 {} {}                   {}{}\n".format(
            address, address + 1, device, signature[1] + inode_delta, mapped, suffix,
        )
        with mock.patch("builtins.open", mock.mock_open(read_data=mapping)):
            compiler._verify_loaded_library(lib, signature, self.library)

    @unittest.skipUnless(compiler.sys.platform.startswith("linux"), "Linux loader mapping check")
    def test_loader_mapping_matches_actual_artifact(self):
        self.verify_mapping()

    @unittest.skipUnless(compiler.sys.platform.startswith("linux"), "Linux loader mapping check")
    def test_loader_reusing_an_old_inode_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "older Y compiler library"):
            self.verify_mapping(inode_delta=1)

    @unittest.skipUnless(compiler.sys.platform.startswith("linux"), "Linux loader mapping check")
    def test_loader_mapping_of_a_deleted_file_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "older Y compiler library"):
            self.verify_mapping(suffix=" (deleted)")

    @unittest.skipUnless(compiler.sys.platform.startswith("linux"), "Linux loader mapping check")
    def test_loader_mapping_of_another_path_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "older Y compiler library"):
            self.verify_mapping(path="/old/liby.so")

    def test_clear_cache_preserves_public_statistics(self):
        self.compile()
        self.compile()
        self.assertEqual(compiler.clear_disk_cache(), 1)
        stats = compiler.get_cache_stats()
        self.assertEqual(stats["mem_hits"], 0)
        self.assertEqual(stats["disk_hits"], 0)
        self.assertEqual(stats["misses"], 0)
        self.assertEqual(stats["cached_files_count"], 0)
        self.assertEqual(stats["cache_dir"], str(self.cache))


class TestRealLibraryMapping(unittest.TestCase):
    """The identity check against a library the dynamic loader really mapped.

    Every check above feeds the parser a synthesized mapping line. One built
    from `stat()`'s device number agreed with a check comparing device numbers,
    and both were wrong on btrfs: there the loader's mapping names the
    superblock's device, so every real library was refused as stale.
    """

    @unittest.skipUnless(compiler.sys.platform.startswith("linux"), "Linux loader mapping check")
    def test_the_freshly_built_library_is_accepted_where_it_lives(self):
        try:
            path = Path(compiler._find_liby()).resolve()
        except (OSError, RuntimeError) as error:
            self.skipTest(f"no built liby.so to load: {error}")
        library = ctypes.CDLL(str(path))
        compiler._verify_loaded_library(library, compiler._library_signature(path), path)


if __name__ == "__main__":
    unittest.main()
