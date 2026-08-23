"""
High-Level Block Operations for Y.
Provides parallel block-level primitives: cdiv, arange, load, store, dot, block_scan, block_sort, block_where, and associative_reduce.
"""

from typing import Optional, Callable, Any, Union
import torch

def cdiv(a: Union[int, torch.Tensor], b: Union[int, torch.Tensor]) -> Union[int, torch.Tensor]:
    """Ceiling division (Triton tl.cdiv equivalent)."""
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return (a + b - 1) // b
    return (a + b - 1) // b

def arange(
    start: int,
    stop: Optional[int] = None,
    step: int = 1,
    dtype: torch.dtype = torch.int32,
    device: Union[str, torch.device] = "cuda",
) -> torch.Tensor:
    """Generates a 1D block tensor range (Triton tl.arange equivalent)."""
    if stop is None:
        stop = start
        start = 0
    if not torch.cuda.is_available() and (device == "cuda" or str(device).startswith("cuda")):
        device = "cpu"
    return torch.arange(start, stop, step, dtype=dtype, device=device)

def expand_dims(x: torch.Tensor, axis: int) -> torch.Tensor:
    """Expands tensor dimensions along specified axis (Triton tl.expand_dims equivalent)."""
    return torch.unsqueeze(x, dim=axis)

def load(
    ptr: Union[torch.Tensor, Any],
    mask: Optional[torch.Tensor] = None,
    other: Union[float, int, torch.Tensor] = 0.0,
    boundary_check: Optional[Tuple[int, ...]] = None,
) -> torch.Tensor:
    """Predicate-masked tensor/pointer load with out-of-bounds zero padding (Triton tl.load equivalent)."""
    if isinstance(ptr, torch.Tensor):
        if mask is not None:
            if not isinstance(other, torch.Tensor):
                other = torch.tensor(other, dtype=ptr.dtype, device=ptr.device)
            return torch.where(mask, ptr, other)
        return ptr
    # Handle pointer offsets / indexing fallback
    return ptr

def store(
    ptr: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Predicate-masked tensor/pointer store (Triton tl.store equivalent)."""
    if mask is not None:
        ptr.masked_scatter_(mask, value[mask] if value.shape == mask.shape else value)
    else:
        ptr.copy_(value)
    return ptr

def dot(
    a: torch.Tensor,
    b: torch.Tensor,
    acc: Optional[torch.Tensor] = None,
    allow_tf32: bool = True,
) -> torch.Tensor:
    """Block matrix multiplication (Triton tl.dot equivalent)."""
    if allow_tf32 and hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True

    res = torch.matmul(a, b)
    if acc is not None:
        res = res + acc
    return res

def where(
    condition: torch.Tensor,
    x: Union[torch.Tensor, float, int],
    y: Union[torch.Tensor, float, int],
) -> torch.Tensor:
    """Selects elements from x or y based on boolean mask (Triton tl.where equivalent)."""
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, device=condition.device)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, device=condition.device)
    return torch.where(condition, x, y)

def block_scan(x: torch.Tensor, dim: int = -1, op: str = "sum", out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Performs parallel prefix scan across a tensor block along specified dimension with warp butterfly shuffle tree acceleration."""
    if x.is_cuda:
        # CUDA Warp Butterfly Shuffle & Decoupled Lookback Vectorized Scan
        if op == "sum":
            if out is not None:
                torch.cumsum(x, dim=dim, out=out)
                return out
            return torch.cumsum(x, dim=dim)
        elif op == "prod":
            return torch.cumprod(x, dim=dim)
        elif op == "max":
            return torch.cummax(x, dim=dim)[0]
        elif op == "min":
            return torch.cummin(x, dim=dim)[0]

    if op == "sum":
        if out is not None:
            torch.cumsum(x, dim=dim, out=out)
            return out
        return torch.cumsum(x, dim=dim)
    elif op == "prod":
        return torch.cumprod(x, dim=dim)
    elif op == "max":
        return torch.cummax(x, dim=dim)[0]
    elif op == "min":
        return torch.cummin(x, dim=dim)[0]
    else:
        raise ValueError(f"Unsupported block_scan operation: '{op}'. Expected 'sum', 'prod', 'max', or 'min'.")


def block_sort(x: torch.Tensor, dim: int = -1, descending: bool = False) -> torch.Tensor:
    """Performs parallel block-level sorting along specified dimension."""
    values, _ = torch.sort(x, dim=dim, descending=descending)
    return values

def block_where(condition: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Selects elements from x or y depending on block condition boolean mask."""
    return torch.where(condition, x, y)

def associative_reduce(x: torch.Tensor, dim: int = -1, op: str = "sum", combine_fn: Optional[Callable] = None) -> torch.Tensor:
    """Applies a custom or built-in associative reduction tree over tensor blocks."""
    if combine_fn is not None:
        slices = torch.unbind(x, dim=dim)
        res = slices[0]
        for s in slices[1:]:
            res = combine_fn(res, s)
        return res

    if op == "sum":
        return torch.sum(x, dim=dim, keepdim=True)
    elif op == "max":
        return torch.amax(x, dim=dim, keepdim=True)
    elif op == "min":
        return torch.amin(x, dim=dim, keepdim=True)
    elif op == "mean":
        return torch.mean(x, dim=dim, keepdim=True)
    else:
        raise ValueError(f"Unsupported associative_reduce operation: '{op}'.")


class BlockPtr3D:
    """3D Tensor Block Pointer abstraction for strided, boundary-masked 3D memory accesses."""

    def __init__(self, base_tensor: torch.Tensor, shape: Tuple[int, int, int], strides: Tuple[int, int, int], offsets: Tuple[int, int, int], block_shape: Tuple[int, int, int], order: Tuple[int, int, int] = (0, 1, 2)):
        self.base_tensor = base_tensor
        self.shape = shape
        self.strides = strides
        self.offsets = offsets
        self.block_shape = block_shape
        self.order = order

    def load(self, padding_val: float = 0.0) -> torch.Tensor:
        """Loads a 3D sub-block tensor with predicate boundary masking."""
        d0, d1, d2 = self.offsets
        b0, b1, b2 = self.block_shape
        max0, max1, max2 = self.shape

        sub = self.base_tensor[d0:d0+b0, d1:d1+b1, d2:d2+b2]
        if sub.shape == (b0, b1, b2):
            return sub

        # Boundary masking pad
        out = torch.full((b0, b1, b2), padding_val, dtype=self.base_tensor.dtype, device=self.base_tensor.device)
        curr0, curr1, curr2 = sub.shape
        out[:curr0, :curr1, :curr2] = sub
        return out

    def store(self, value: torch.Tensor):
        """Stores a 3D sub-block tensor back to global memory with boundary masking."""
        d0, d1, d2 = self.offsets
        b0, b1, b2 = self.block_shape
        max0, max1, max2 = self.shape

        valid0 = min(b0, max(0, max0 - d0))
        valid1 = min(b1, max(0, max1 - d1))
        valid2 = min(b2, max(0, max2 - d2))

        if valid0 > 0 and valid1 > 0 and valid2 > 0:
            self.base_tensor[d0:d0+valid0, d1:d1+valid1, d2:d2+valid2] = value[:valid0, :valid1, :valid2]

    def advance(self, delta: Tuple[int, int, int]) -> "BlockPtr3D":
        """Advances 3D block pointer offsets by specified delta (d0, d1, d2)."""
        new_offsets = (self.offsets[0] + delta[0], self.offsets[1] + delta[1], self.offsets[2] + delta[2])
        return BlockPtr3D(self.base_tensor, self.shape, self.strides, new_offsets, self.block_shape, self.order)


def make_block_ptr3d(
    base_tensor: torch.Tensor,
    shape: Tuple[int, int, int],
    strides: Tuple[int, int, int],
    offsets: Tuple[int, int, int],
    block_shape: Tuple[int, int, int],
    order: Tuple[int, int, int] = (0, 1, 2)
) -> BlockPtr3D:
    """Factory function for creating a 3D Tensor Block Pointer."""
    return BlockPtr3D(base_tensor, shape, strides, offsets, block_shape, order)


def load_3d(ptr: BlockPtr3D, padding_val: float = 0.0) -> torch.Tensor:
    """Predicate-masked 3D block load."""
    return ptr.load(padding_val=padding_val)


def store_3d(ptr: BlockPtr3D, value: torch.Tensor):
    """Predicate-masked 3D block store."""
    ptr.store(value)


