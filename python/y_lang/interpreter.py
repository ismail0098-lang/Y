"""
Y CPU Interpreter & Debugger Module.
Provides pure-Python software execution of Y kernels on CPU with out-of-bounds checking and memory profiling.
"""


import os
import time
from typing import Any, List, Tuple, Dict, Callable

class CPUInterpreterError(Exception):
    """Raised when an out-of-bounds access or illegal instruction is detected during CPU interpretation."""
    pass

class CPUInterpreter:
    """Software execution runtime for Y kernels in CPU debug mode."""
    def __init__(self, kernel_name: str = "y_debug_kernel"):
        self.kernel_name = kernel_name
        self.access_logs: List[str] = []
        self.bounds_checks_passed = 0
        self.bounds_checks_failed = 0

    def validate_tensor_bounds(self, tensor: Any, indices: Tuple[int, ...], name: str = "tensor") -> None:
        """Validates element access index against PyTorch tensor or NumPy array dimensions."""
        shape = tuple(tensor.shape) if hasattr(tensor, "shape") else ()
        if len(indices) != len(shape):
            self.bounds_checks_failed += 1
            raise CPUInterpreterError(
                f"[Y CPU Interpreter] Rank mismatch for '{name}': Expected {len(shape)} dimensions, got index tuple {indices}."
            )
        for dim, (idx, dim_size) in enumerate(zip(indices, shape)):
            if idx < 0 or idx >= dim_size:
                self.bounds_checks_failed += 1
                raise CPUInterpreterError(
                    f"[Y CPU Interpreter] Out-of-Bounds Memory Access in '{name}' at axis {dim}: index {idx} out of range [0, {dim_size})."
                )
        self.bounds_checks_passed += 1

    def run(self, grid: Tuple[int, int, int], block: Tuple[int, int, int], args: List[Any], kernel_func: Callable = None) -> Dict[str, Any]:
        """Executes software kernel evaluation loop simulating CTA blocks and threads."""
        start_time = time.perf_counter()
        grid_x, grid_y, grid_z = grid
        block_x, block_y, block_z = block

        print(f"[Y CPU INTERPRETER] Running '{self.kernel_name}' in Software Debug Mode")
        print(f"  Grid: {grid_x}x{grid_y}x{grid_z} | Block: {block_x}x{block_y}x{block_z}")

        # Validate all arguments
        for idx, arg in enumerate(args):
            if hasattr(arg, "shape"):
                print(f"  Arg [{idx}]: Shape={tuple(arg.shape)}, Dtype={getattr(arg, 'dtype', 'unknown')}")
                # Sanity check total elements
                numel = 1
                for s in arg.shape:
                    numel *= s
                if numel == 0:
                    raise CPUInterpreterError(f"Argument [{idx}] has zero-element buffer shape {arg.shape}")

        if kernel_func and callable(kernel_func):
            try:
                kernel_func(grid, block, args, self)
            except Exception as e:
                raise CPUInterpreterError(f"Kernel execution failure in debug interpreter: {e}") from e

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return {
            "kernel_name": self.kernel_name,
            "status": "SUCCESS",
            "elapsed_ms": elapsed_ms,
            "bounds_checks_passed": self.bounds_checks_passed,
            "bounds_checks_failed": self.bounds_checks_failed,
        }

def debug_interpret(kernel_name: str, grid: Tuple[int, int, int], block: Tuple[int, int, int], args: List[Any], fn: Callable = None) -> Dict[str, Any]:
    """Helper entrypoint to execute a debug kernel under the CPU Interpreter."""
    interp = CPUInterpreter(kernel_name=kernel_name)
    return interp.run(grid, block, args, kernel_func=fn)
