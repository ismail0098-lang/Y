"""
`torch.compile` backend that lowers elementwise float32 subgraphs to Y kernels.

    compiled = torch.compile(model, backend=y_inductor)

What is lowered: connected groups of `add`, `sub`, `mul` and `relu` whose every
tensor operand and result is a contiguous float32 CUDA tensor of one static
shape (no broadcasting). A tensor-scalar `add`, `sub` or `mul` carries the
scalar in the kernel. Each group becomes one Y kernel, compiled through the Y
compiler to PTX and launched on the current CUDA stream.

Everything else runs exactly as PyTorch runs it: `linear`, `matmul`, division,
negation, `exp`, `sigmoid`, `silu`, reductions, broadcasting, other dtypes,
CPU tensors, and any call that needs autograd. `compiled.y_report` records
which nodes were lowered, why each other node was not, and, per kernel, how
many calls launched it and how many fell back to the original subgraph.

Every lowered result is bit-for-bit what eager PyTorch computes. That is a
constraint on the design, not luck, and it decides the op set:

* **A multiply never feeds an add or a subtract inside one kernel.** ptxas
  contracted that pair into one fused multiply-add, rounding once where eager
  rounds twice: a Y kernel computing `let p = a * b; let s = p + c;` differed
  from eager `(a * b) + c` in 245,999 of 1,048,576 results. Since 2026-10-11
  the PTX emitter rounds a `let`-bound product first (`mul.rn`), and the same
  kernel matches eager in every result, so this rule is now conservative:
  the `let`-per-node source this module writes would be exact if fused. It
  stays until fusing the pair is measured.
* **Negation is not lowered.** Eager `-x` canonicalises a NaN to 0x7fffffff; a
  negation ptxas folds into a select keeps the payload and flips its sign.
* **Division is not lowered.** Y's F32 `/` is `div.approx.f32`; eager's is
  correctly rounded.
* **`relu` is lowered as `x <= 0.0 ? 0.0 : x`**, which measures bit-identical
  to `torch.relu` on CUDA including NaN payloads (kept, sign included) and
  `-0.0` (becomes `+0.0`).

This module used to lower nothing. It returned the original graph under
`torch.no_grad()`, so `torch.compile(model, backend=y_inductor)` computed eager
PyTorch, reported a fused-cluster count for kernels that did not exist, and
broke training: every output had `requires_grad=False` and `backward()` raised.
"""

import math
import operator
import struct
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.fx
from torch.fx.passes.utils.fuser_utils import fuse_by_partitions

ADD, SUB, MUL, RELU = "add", "sub", "mul", "relu"
_BINARY_SYMBOL = {ADD: "+", SUB: "-", MUL: "*"}

_FUNCTIONS = {
    operator.add: ADD,
    torch.add: ADD,
    operator.sub: SUB,
    torch.sub: SUB,
    operator.mul: MUL,
    torch.mul: MUL,
    torch.relu: RELU,
    torch.nn.functional.relu: RELU,
}
_METHODS = {"add": ADD, "sub": SUB, "mul": MUL, "relu": RELU}

# Ops this backend declines, with the reason stated where a user will see it.
_NEGATION = ("negation is not lowered: eager `-x` canonicalises a NaN, and a negation "
             "ptxas folds into a select keeps its payload and flips its sign")
_DIVISION = ("division is not lowered: Y's F32 `/` is `div.approx.f32`, and eager "
             "division is correctly rounded")
_DECLINED = {
    operator.neg: _NEGATION,
    torch.neg: _NEGATION,
    operator.truediv: _DIVISION,
    torch.div: _DIVISION,
    torch.true_divide: _DIVISION,
}
_DECLINED_METHODS = {"neg": _NEGATION, "div": _DIVISION, "true_divide": _DIVISION}

# `tid` is a 32-bit index: the last thread of the last block must not overflow.
_MAX_NUMEL = 2**31 - 256
_BLOCK = 256
# An integer scalar is used only where float32 holds it exactly, so there is no
# question which way an int64 -> float rounding goes.
_MAX_EXACT_INT = 2**24


def _describe(node: torch.fx.Node) -> str:
    t = node.target
    if node.op == "call_method" or isinstance(t, str):
        return f"{node.op} `{t}`"
    return f"{node.op} `{getattr(t, '__name__', str(t))}`"


def _tensor_info(node: Any) -> Optional[Tuple[Tuple[int, ...], torch.dtype, torch.device, bool]]:
    """(shape, dtype, device, contiguous) of a node's value, or None if the
    node is not a tensor with a static shape."""
    if not isinstance(node, torch.fx.Node):
        return None
    v = node.meta.get("val", node.meta.get("example_value"))
    if not isinstance(v, torch.Tensor):
        return None
    shape = tuple(v.shape)
    if not all(isinstance(d, int) for d in shape):
        return None
    return shape, v.dtype, v.device, v.is_contiguous()


def _scalar_literal(value: Any) -> Tuple[Optional[str], Optional[str]]:
    """The exact Y literal for a Python scalar operand, or a reason it is not
    lowered. The literal is the exact decimal of the float32 the scalar
    becomes, so reading it back as a double and narrowing to float32 is exact."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, f"operand {value!r} is neither a tensor nor a number"
    if isinstance(value, int) and abs(value) > _MAX_EXACT_INT:
        return None, f"integer scalar {value} is not exact in float32"
    x = float(value)
    if not math.isfinite(x):
        return None, f"scalar {value!r} is not finite"
    try:
        f32 = struct.unpack("<f", struct.pack("<f", x))[0]
    except OverflowError:
        return None, f"scalar {value!r} overflows float32"
    text = format(Decimal(abs(f32)), "f")
    if "." not in text:
        text += ".0"
    if math.copysign(1.0, f32) < 0:
        return f"(-{text})", None
    return text, None


def _kind(gm: torch.fx.GraphModule, node: torch.fx.Node) -> Tuple[Optional[str], Optional[str]]:
    if node.op == "call_function":
        if node.target in _DECLINED:
            return None, _DECLINED[node.target]
        kind = _FUNCTIONS.get(node.target)
    elif node.op == "call_method":
        if node.target in _DECLINED_METHODS:
            return None, _DECLINED_METHODS[node.target]
        kind = _METHODS.get(node.target)
    elif node.op == "call_module":
        mod = gm.get_submodule(node.target)
        kind = RELU if type(mod) is torch.nn.ReLU and not mod.inplace else None
    else:
        return None, None  # not a computation: nothing to lower or report
    if kind is None:
        return None, f"{_describe(node)} is not lowered"
    return kind, None


def _classify(gm: torch.fx.GraphModule, node: torch.fx.Node) -> Tuple[Optional[str], Optional[str]]:
    kind, reason = _kind(gm, node)
    if kind is None:
        return None, reason
    kwargs = dict(node.kwargs)
    if kind == RELU and kwargs.get("inplace", False) is False:
        kwargs.pop("inplace", None)
    if kwargs:
        return None, f"keyword arguments {sorted(kwargs)} are not lowered"
    arity = 1 if kind == RELU else 2
    if len(node.args) != arity:
        return None, f"{arity} positional operand(s) expected, found {len(node.args)}"

    info = _tensor_info(node)
    if info is None:
        return None, "the result has no static tensor shape"
    shape, dtype, device, contiguous = info
    if dtype != torch.float32:
        return None, f"dtype {dtype} (only float32 is lowered)"
    if device.type != "cuda":
        return None, f"device {device} (only CUDA is lowered)"
    if not contiguous:
        return None, "the result is not contiguous"
    numel = math.prod(shape)
    if numel == 0 or numel > _MAX_NUMEL:
        return None, f"{numel} elements (1 to {_MAX_NUMEL} are lowered)"

    tensors = 0
    for arg in node.args:
        if isinstance(arg, torch.fx.Node):
            a = _tensor_info(arg)
            if a is None:
                return None, f"operand `{arg.name}` is not a tensor with a static shape"
            if a != info:
                return None, (f"operand `{arg.name}` is {a[0]} {a[1]} on {a[2]}, the result is "
                              f"{shape} {dtype} on {device} (broadcasting and mixed operands "
                              "are not lowered)")
            tensors += 1
        else:
            if kind == RELU:
                return None, "relu of a scalar is not lowered"
            _, why = _scalar_literal(arg)
            if why is not None:
                return None, why
            if kind == MUL and float(arg) == 1.0:
                # Measured: ptxas deletes `mul.f32 x, 1.0` (the SASS stores
                # the loaded register), so a NaN input keeps its payload where
                # eager's multiply canonicalises it to 0x7fffffff.
                return None, ("a multiply by 1.0 is not lowered: ptxas deletes it, so a NaN "
                              "keeps its payload where eager's multiply canonicalises it")
    if tensors == 0:
        return None, "no tensor operand"
    return kind, None


def _fusable(producer_kind: str, consumer_kind: str) -> bool:
    # ptxas contracted a multiply feeding an add or a subtract into one FMA,
    # rounding once where eager rounds twice. The emitter now rounds the
    # `let`-bound product first, so this is conservative (module docstring).
    return not (producer_kind == MUL and consumer_kind in (ADD, SUB))


def _partition(gm: torch.fx.GraphModule, kinds: Dict[torch.fx.Node, str]) -> List[List[torch.fx.Node]]:
    """Greedy grouping in graph order, acyclic between groups by construction.

    A node joins a producer's group only when every producer already in that
    group feeds it through a fusable edge, and the group is not an ancestor of
    any producer outside it - otherwise the fused kernel would need its own
    output. The ancestry is tracked between GROUPS, not nodes: a group runs as
    one kernel, so it depends on everything any member depends on, and a check
    between nodes misses a cycle through two groups."""
    group_of: Dict[torch.fx.Node, int] = {}
    groups: List[List[torch.fx.Node]] = []
    preds: List[set] = []  # preds[g]: the groups g directly depends on
    # The groups a node depends on through paths that meet no other group.
    reach: Dict[torch.fx.Node, set] = {}

    def contribution(p: torch.fx.Node) -> set:
        g = group_of.get(p)
        return {g} if g is not None else reach.get(p, set())

    def is_ancestor(target: int, start: set) -> bool:
        stack, seen = list(start), set()
        while stack:
            g = stack.pop()
            if g == target:
                return True
            if g not in seen:
                seen.add(g)
                stack.extend(preds[g])
        return False

    for node in gm.graph.nodes:
        producers = node.all_input_nodes
        if node not in kinds:
            reach[node] = set().union(*(contribution(p) for p in producers))
            continue
        chosen = None
        for p in producers:
            g = group_of.get(p)
            if g is None:
                continue
            inside = [q for q in producers if group_of.get(q) == g]
            if not all(_fusable(kinds[q], kinds[node]) for q in inside):
                continue
            outside = set().union(*(contribution(q) for q in producers if group_of.get(q) != g))
            if is_ancestor(g, outside):
                continue
            chosen = g
            preds[g] |= outside
            break
        if chosen is None:
            chosen = len(groups)
            groups.append([])
            preds.append(set().union(*(contribution(p) for p in producers)))
        group_of[node] = chosen
        groups[chosen].append(node)
    return groups


def _kernel_source(sub: torch.fx.GraphModule) -> Tuple[str, int, int, List[str]]:
    """Y source for one fused subgraph: one thread per element, one `let` per
    node. Returns (source, number of inputs, number of outputs, op list)."""
    env: Dict[torch.fx.Node, str] = {}
    params: List[str] = []
    body: List[str] = []
    ops: List[str] = []
    n_in = 0
    counter = 0

    def fresh() -> str:
        nonlocal counter
        counter += 1
        return f"v{counter}"

    def operand(a: Any) -> str:
        if isinstance(a, torch.fx.Node):
            return env[a]
        text, why = _scalar_literal(a)
        assert why is None, why
        return text

    outputs: Tuple[torch.fx.Node, ...] = ()
    for node in sub.graph.nodes:
        if node.op == "placeholder":
            params.append(f"x{n_in}: GlobalMemory<F32>")
            v = fresh()
            body.append(f"        let {v}: F32 = x{n_in}[tid];")
            env[node] = v
            n_in += 1
        elif node.op == "output":
            outputs = node.args[0]
        else:
            kind, why = _kind(sub, node)
            assert kind is not None, why
            ops.append(kind)
            v = fresh()
            if kind == RELU:
                a = operand(node.args[0])
                body.append(f"        let mut {v}: F32 = {a};")
                body.append(f"        if {a} <= 0.0 {{")
                body.append(f"            {v} = 0.0;")
                body.append("        }")
            else:
                a, b = operand(node.args[0]), operand(node.args[1])
                body.append(f"        let {v}: F32 = {a} {_BINARY_SYMBOL[kind]} {b};")
            env[node] = v
    for j, out in enumerate(outputs):
        params.append(f"y{j}: GlobalMemory<F32>")
        body.append(f"        y{j}[tid] = {env[out]};")
    source = (
        f"kernel y_kernel({', '.join(params)}, n: I32) {{\n"
        "    let tid: I32 = block_idx_x() * block_dim_x() + thread_idx_x();\n"
        "    if tid < n {\n" + "\n".join(body) + "\n    }\n}\nfn main() {}\n"
    )
    return source, n_in, len(outputs), ops


class LoweredKernel:
    """What one Y kernel computes, and what happened when it was called."""

    def __init__(self, name: str, nodes: List[str], ops: List[str], source: str):
        self.name = name
        self.nodes = nodes
        self.ops = ops
        self.source = source
        self.launches = 0
        self.fallbacks: Dict[str, int] = {}

    def __repr__(self) -> str:
        return (f"LoweredKernel({self.name}: {' '.join(self.ops)}; launches={self.launches}, "
                f"fallbacks={self.fallbacks})")


class YLoweringReport:
    def __init__(self):
        self.kernels: List[LoweredKernel] = []
        self.not_lowered: List[Tuple[str, str, str]] = []  # (node, what, why)

    @property
    def lowered_ops(self) -> int:
        return sum(len(k.ops) for k in self.kernels)

    def summary(self) -> str:
        lines = [f"y_inductor: {len(self.kernels)} Y kernel(s), {self.lowered_ops} op(s) lowered"]
        for k in self.kernels:
            lines.append(f"  {k.name}: {' -> '.join(k.ops)} | launches {k.launches}, "
                         f"fallbacks {k.fallbacks}")
        for node, what, why in self.not_lowered:
            lines.append(f"  not lowered: {node} ({what}): {why}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


class _YKernelModule(torch.nn.Module):
    """Runs one fused subgraph as a Y kernel, or as the original subgraph when
    this call cannot use the kernel."""

    def __init__(self, fallback: torch.fx.GraphModule, source: str, n_in: int, n_out: int,
                 shape: Tuple[int, ...], device: torch.device, record: LoweredKernel):
        super().__init__()
        from .torch_interop import TorchKernel

        self.fallback = fallback
        self.kernel = TorchKernel(source, kernel_name="y_kernel")  # compiles now
        self.n_in = n_in
        self.n_out = n_out
        self.shape = shape
        self.numel = math.prod(shape)
        self.device = device
        self.record = record

    def _why_not(self, args: Tuple[Any, ...]) -> Optional[str]:
        if len(args) != self.n_in:
            return f"{len(args)} inputs, compiled for {self.n_in}"
        if torch.is_grad_enabled() and any(isinstance(a, torch.Tensor) and a.requires_grad
                                           for a in args):
            return "autograd"
        for a in args:
            if not isinstance(a, torch.Tensor):
                return "a non-tensor input"
            if a.dtype != torch.float32 or a.device != self.device or tuple(a.shape) != self.shape:
                return "an input differs from the compiled shape, dtype or device"
            if not a.is_contiguous():
                return "a non-contiguous input"
        if self.device.index is not None and self.device.index != torch.cuda.current_device():
            return "the tensors are not on the current CUDA device"
        return None

    def forward(self, *args):
        why = self._why_not(args)
        if why is not None:
            self.record.fallbacks[why] = self.record.fallbacks.get(why, 0) + 1
            return self.fallback(*args)
        outs = tuple(torch.empty(self.shape, dtype=torch.float32, device=self.device)
                     for _ in range(self.n_out))
        grid = ((self.numel + _BLOCK - 1) // _BLOCK, 1, 1)
        self.kernel.launch(grid, (_BLOCK, 1, 1), list(args) + list(outs) + [self.numel])
        self.record.launches += 1
        return outs


def _propagate_shapes(gm: torch.fx.GraphModule, example_inputs: List[Any]) -> Optional[str]:
    """Fills `node.meta['val']` with fake tensors unless Dynamo already did.
    Fake, not real: running the graph for its shapes would consume RNG state
    and do the work twice. Returns a reason on failure."""
    computing = [n for n in gm.graph.nodes
                 if n.op in ("call_function", "call_method", "call_module")]
    if all(("val" in n.meta or "example_value" in n.meta) for n in computing):
        return None
    try:
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.passes.fake_tensor_prop import FakeTensorProp

        FakeTensorProp(gm, mode=FakeTensorMode(allow_non_fake_inputs=True)).propagate(*example_inputs)
    except Exception as e:  # the graph still runs; nothing is lowered, and the report says why
        return f"shape propagation failed: {type(e).__name__}: {e}"
    return None


def y_inductor(gm: Any, example_inputs: List[torch.Tensor]) -> Callable:
    """`torch.compile` backend: lowers elementwise float32 CUDA subgraphs to Y
    kernels and leaves everything else as it was. Also accepts an `nn.Module`,
    which is symbolically traced first. The result carries `.y_report`."""
    if not isinstance(gm, torch.fx.GraphModule):
        gm = torch.fx.symbolic_trace(gm)
    report = YLoweringReport()

    failure = _propagate_shapes(gm, list(example_inputs))
    kinds: Dict[torch.fx.Node, str] = {}
    for node in gm.graph.nodes:
        if failure is not None:
            if node.op in ("call_function", "call_method", "call_module"):
                report.not_lowered.append((node.name, _describe(node), failure))
            continue
        kind, why = _classify(gm, node)
        if kind is not None:
            kinds[node] = kind
        elif why is not None:
            report.not_lowered.append((node.name, _describe(node), why))

    groups = _partition(gm, kinds)
    # What each group is, read before fusing erases its nodes.
    described = [([n.name for n in g], _tensor_info(g[0])) for g in groups]
    if groups:
        # One call for every group: `fuse_by_partitions` fuses them in turn and
        # re-validates each against the graph as it stands (raising on a
        # cycle, which `_partition` rules out), then sorts the graph once.
        gm = fuse_by_partitions(gm, [dict.fromkeys(g) for g in groups], prefix="y_kernel_",
                                always_return_tuple=True)
    for i, (names, info) in enumerate(described):
        name = f"y_kernel_{i}"
        sub = getattr(gm, name)
        shape, _, device, _ = info
        source, n_in, n_out, ops = _kernel_source(sub)
        record = LoweredKernel(name, names, ops, source)
        setattr(gm, name, _YKernelModule(sub, source, n_in, n_out, shape, device, record))
        report.kernels.append(record)
    gm.recompile()
    gm.y_report = report
    return gm
