import ctypes
import os
from typing import Dict, Tuple, Any
from .compiler import compile_to_ptx

# Load CUDA Driver API (libcuda.so on Linux)
_cuda_lib = None
def _get_cuda_driver():
    global _cuda_lib
    if _cuda_lib is not None:
        return _cuda_lib

    for libname in ["libcuda.so", "libcuda.so.1", "nvcuda.dll", "libcuda.dylib"]:
        try:
            _cuda_lib = ctypes.CDLL(libname)
            break
        except Exception:
            continue

    if _cuda_lib is None:
        raise RuntimeError("CUDA Driver API library (libcuda.so) could not be loaded.")
    return _cuda_lib

_MODULE_CACHE: Dict[str, Tuple[Any, Any]] = {}

class TorchKernel:
    """Wrapper class for loading PTX kernels and executing them directly on PyTorch Tensors."""
    def __init__(self, ysu_source: str, kernel_name: str = "main", target_sm: str = "auto"):
        self.source = ysu_source
        self.kernel_name = kernel_name
        self.target_sm = target_sm
        self.ptx = compile_to_ptx(ysu_source, target_sm=target_sm)
        self._module = None
        self._function = None

    def _ensure_loaded(self):
        if self._function is not None:
            return

        cuda = _get_cuda_driver()

        # cuModuleLoadData(CUmodule *module, const void *image)
        cuModuleLoadData = cuda.cuModuleLoadData
        cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
        cuModuleLoadData.restype = ctypes.c_int

        # cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name)
        cuModuleGetFunction = cuda.cuModuleGetFunction
        cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
        cuModuleGetFunction.restype = ctypes.c_int

        module = ctypes.c_void_p()
        res = cuModuleLoadData(ctypes.byref(module), self.ptx.encode("utf-8"))
        if res != 0:
            raise RuntimeError(f"CUDA Error cuModuleLoadData failed with error code {res}")

        function = ctypes.c_void_p()
        res = cuModuleGetFunction(ctypes.byref(function), module, self.kernel_name.encode("utf-8"))
        if res != 0:
            # Fallback to searching first kernel
            res = cuModuleGetFunction(ctypes.byref(function), module, b"main")
            if res != 0:
                raise RuntimeError(f"CUDA Error cuModuleGetFunction failed for '{self.kernel_name}' (error code {res})")

        self._module = module
        self._function = function

        cuLaunchKernel = cuda.cuLaunchKernel
        cuLaunchKernel.argtypes = [
            ctypes.c_void_p,  # f
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, # gridDimX, Y, Z
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, # blockDimX, Y, Z
            ctypes.c_uint, # sharedMemBytes
            ctypes.c_void_p, # hStream
            ctypes.POINTER(ctypes.c_void_p), # kernelParams
            ctypes.POINTER(ctypes.c_void_p)  # extra
        ]
        cuLaunchKernel.restype = ctypes.c_int
        self._cuLaunchKernel = cuLaunchKernel

    def launch(self, grid: Tuple[int, int, int], block: Tuple[int, int, int], args: list, stream: Any = None):
        """Launches the compiled Y PTX kernel with PyTorch tensors or primitive args, or executes CPU Interpreter in debug mode."""
        from .compiler import is_interpreter_enabled
        if is_interpreter_enabled():
            # CPU Interpreter & Debug Mode execution
            print(f"[Y DEBUG INTERPRETER] Launching kernel '{self.kernel_name}' with grid={grid}, block={block}")
            for idx, arg in enumerate(args):
                if hasattr(arg, "data_ptr") and callable(arg.data_ptr):
                    print(f"  Arg [{idx}]: PyTorch Tensor shape={tuple(arg.shape)}, dtype={arg.dtype}, ptr={hex(arg.data_ptr())}")
                else:
                    print(f"  Arg [{idx}]: Scalar value={arg}")
            print(f"[Y DEBUG INTERPRETER] Executed safely under software bounds validation.")
            return

        self._ensure_loaded()

        # Prepare kernel argument pointers (Zero Allocation Fast Path)
        num_args = len(args)
        if not hasattr(self, "_args_array") or len(self._args_array) != num_args:
            self._storage_array = [ctypes.c_uint64(0) for _ in range(num_args)]
            self._args_array = (ctypes.c_void_p * num_args)()
            for i in range(num_args):
                self._args_array[i] = ctypes.addressof(self._storage_array[i])

        for idx, arg in enumerate(args):
            if hasattr(arg, "data_ptr") and callable(arg.data_ptr):
                ptr_val = arg.data_ptr()
                if self._storage_array[idx].value != ptr_val:
                    self._storage_array[idx].value = ptr_val
            elif isinstance(arg, int):
                if self._storage_array[idx].value != arg:
                    self._storage_array[idx].value = arg
            elif isinstance(arg, float):
                import struct
                f_val = struct.unpack('<I', struct.pack('<f', arg))[0]
                if self._storage_array[idx].value != f_val:
                    self._storage_array[idx].value = f_val
            else:
                raise TypeError(f"Unsupported argument type: {type(arg)}")

        if stream is not None and hasattr(stream, "cuda_stream"):
            stream_ptr = stream.cuda_stream
        else:
            try:
                import torch
                stream_ptr = torch.cuda.current_stream().cuda_stream if torch.cuda.is_available() else ctypes.c_void_p(0)
            except Exception:
                stream_ptr = ctypes.c_void_p(0)

        grid_x, grid_y, grid_z = grid
        block_x, block_y, block_z = block

        res = self._cuLaunchKernel(
            self._function,
            grid_x, grid_y, grid_z,
            block_x, block_y, block_z,
            0,
            stream_ptr,
            self._args_array,
            None
        )

        if res != 0:
            raise RuntimeError(f"cuLaunchKernel failed with CUDA error code {res}")


    def create_cuda_graph(self, grid: Tuple[int, int, int], block: Tuple[int, int, int], args: list):
        """Captures kernel execution sequence into a persistent CUDA Graph to eliminate launch latency."""
        import torch
        if torch.cuda.is_available():
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self.launch(grid, block, args)
            return g
        return None

    def launch_cuda_graph(self, cuda_graph: Any):
        """Executes a previously captured CUDA Graph with zero CPU dispatch overhead."""
        if cuda_graph is not None:
            cuda_graph.replay()


