"""`y_inductor` lowers elementwise float32 CUDA subgraphs to Y kernels, and
every lowered result is bit-for-bit what eager PyTorch computes.

It used to lower nothing. It returned the original graph wrapped in
`torch.no_grad()`, so `torch.compile(model, backend=y_inductor)` ran eager
PyTorch, stamped `_y_compiled=True` and a fused-cluster count on the result,
and broke training: outputs had `requires_grad=False` and `backward()` raised.
`test_training_through_the_backend_keeps_autograd` fails on that
implementation, on the CPU as well as on the GPU.

The GPU tests skip without CUDA and say so. Every test compiles into a PTX
cache directory of its own: the package's disk cache is keyed on the source
text alone, so a shared one would serve PTX from an older compiler build.
"""

import os
import shutil
import tempfile
import unittest

import torch

import y_lang
from y_lang import TorchKernel, y_inductor

CUDA = torch.cuda.is_available()
NO_CUDA = "SKIP: no CUDA device - the lowering itself is not exercised"

# float32 bit patterns that decide bit-for-bit agreement: NaNs with payloads
# and both signs, signed zeros, the smallest denormals, infinities, and the
# largest finite values (so an add overflows).
SPECIAL = [0x7FC0DEAD, 0xFFC0BEEF, 0x80000000, 0x00000000, 0x80000001, 0x00000001,
           0x7F800000, 0xFF800000, 0x3F800000, 0xBF800000, 0x7F7FFFFF, 0xFF7FFFFF, 0x00800000]


def special_values():
    return torch.tensor(SPECIAL, dtype=torch.int64).to(torch.int32).view(torch.float32)


def operands(count, n_random=4096, seed=0, device="cuda"):
    """`count` float32 tensors of one length. The first len(SPECIAL)**2
    positions pair every special value with every other in the first two
    tensors (a third gets a shifted copy); the rest are random normals."""
    s = special_values()
    k = len(s)
    specials = [s.repeat_interleave(k), s.repeat(k), s.roll(3).repeat(k)]
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(count):
        head = specials[i] if i < len(specials) else specials[-1].roll(i)
        out.append(torch.cat([head, torch.randn(n_random, generator=g) * 3]).to(device))
    return out


def bits(t):
    return t.contiguous().view(torch.int32)


def compile_with_reports(model):
    """`torch.compile` with y_inductor, keeping the report of each graph."""
    reports = []

    def backend(gm, example_inputs):
        out = y_inductor(gm, example_inputs)
        reports.append(out.y_report)
        return out

    torch._dynamo.reset()
    return torch.compile(model, backend=backend), reports


class Elementwise(torch.nn.Module):
    def forward(self, x, y, z):
        return torch.relu(x + y) * z - 0.5


class YInductorTest(unittest.TestCase):
    def setUp(self):
        self._cache = tempfile.mkdtemp(prefix="y_inductor_cache_")
        self._old = os.environ.get("YSU_CACHE_DIR")
        os.environ["YSU_CACHE_DIR"] = self._cache
        y_lang.compiler._JIT_CACHE.clear()

    def tearDown(self):
        if self._old is None:
            os.environ.pop("YSU_CACHE_DIR", None)
        else:
            os.environ["YSU_CACHE_DIR"] = self._old
        shutil.rmtree(self._cache, ignore_errors=True)

    def test_training_through_the_backend_keeps_autograd(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(16, 16)

            def forward(self, x):
                return torch.relu(self.lin(x)) + x

        for device in ["cpu"] + (["cuda"] if CUDA else []):
            torch.manual_seed(0)
            model = Net().to(device)
            eager = Net().to(device)
            eager.load_state_dict(model.state_dict())
            x = torch.randn(8, 16, device=device)

            cm, reports = compile_with_reports(model)
            out = cm(x)
            ref = eager(x)
            self.assertTrue(out.requires_grad, f"{device}: the compiled output lost autograd")
            out.sum().backward()
            ref.sum().backward()
            self.assertTrue(torch.equal(bits(out), bits(ref)), device)
            for (name, p), (_, q) in zip(model.named_parameters(), eager.named_parameters()):
                self.assertTrue(torch.equal(bits(p.grad), bits(q.grad)),
                                f"{device}: the gradient of {name} differs from eager")
            if device == "cuda":
                kernels = reports[-1].kernels
                self.assertTrue(kernels, "the elementwise tail should still be lowered")
                self.assertEqual(sum(k.launches for k in kernels), 0,
                                 "a call that needs autograd must not launch a Y kernel")
                self.assertTrue(all(k.fallbacks.get("autograd", 0) >= 1 for k in kernels))

    @unittest.skipUnless(CUDA, NO_CUDA)
    def test_an_elementwise_graph_runs_as_y_kernels_bit_for_bit(self):
        x, y, z = operands(3)
        cm, reports = compile_with_reports(Elementwise())
        with torch.no_grad():
            out = cm(x, y, z)
            ref = Elementwise()(x, y, z)
        kernels = reports[-1].kernels
        self.assertEqual([k.ops for k in kernels], [["add", "relu", "mul"], ["sub"]],
                         "the multiply feeding the subtract must start a new kernel")
        self.assertEqual([k.launches for k in kernels], [1, 1], "every kernel must have run")
        self.assertIn("kernel y_kernel(", kernels[0].source)
        self.assertTrue(torch.equal(bits(out), bits(ref)), "a lowered result differs from eager")

    @unittest.skipUnless(CUDA, NO_CUDA)
    def test_relu_matches_eager_on_every_special_value(self):
        # relu's own result, returned directly: `-0.0` becomes `+0.0` and a NaN
        # keeps its payload, sign included. Every other test feeds relu into an
        # add or a multiply, which erases the sign of a zero - so a relu
        # written as `x < 0.0 ? 0.0 : x` passed all of them.
        class Relus(torch.nn.Module):
            def forward(self, x):
                return torch.relu(x), torch.nn.functional.relu(x)

        (x,) = operands(1)
        cm, reports = compile_with_reports(Relus())
        with torch.no_grad():
            a, b = cm(x)
        self.assertEqual([k.launches for k in reports[-1].kernels], [1, 1])
        ref = torch.relu(x)
        self.assertTrue(torch.equal(bits(a), bits(ref)), "torch.relu differs from eager")
        self.assertTrue(torch.equal(bits(b), bits(ref)), "F.relu differs from eager")

    @unittest.skipUnless(CUDA, NO_CUDA)
    def test_a_multiply_is_never_fused_into_the_add_it_feeds(self):
        n = 1 << 20
        torch.manual_seed(7)
        a, b, c = (torch.randn(n, device="cuda") for _ in range(3))

        class MulAdd(torch.nn.Module):
            def forward(self, a, b, c):
                return a * b + c, c - a * b

        cm, reports = compile_with_reports(MulAdd())
        with torch.no_grad():
            s, d = cm(a, b, c)
        ops = [k.ops for k in reports[-1].kernels]
        for group in ops:
            self.assertFalse("mul" in group and ("add" in group or "sub" in group),
                             f"a multiply shares a kernel with an add or subtract: {ops}")
        self.assertTrue(torch.equal(bits(s), bits(a * b + c)))
        self.assertTrue(torch.equal(bits(d), bits(c - a * b)))

        # The premise, measured: fusing the pair DOES change the answer, because
        # ptxas contracts it into one rounding. If this ever stops differing,
        # the rule above is costing kernels for nothing and should be revisited.
        fused = TorchKernel(
            "kernel y_mad(xa: GlobalMemory<F32>, xb: GlobalMemory<F32>, xc: GlobalMemory<F32>, "
            "out: GlobalMemory<F32>, n: I32) {\n"
            "    let i: I32 = block_idx_x() * block_dim_x() + thread_idx_x();\n"
            "    if i < n {\n        let p: F32 = xa[i] * xb[i];\n        let s: F32 = p + xc[i];\n"
            "        out[i] = s;\n    }\n}\nfn main() {}\n", kernel_name="y_mad")
        out = torch.empty_like(a)
        fused.launch(((n + 255) // 256, 1, 1), (256, 1, 1), [a, b, c, out, n])
        torch.cuda.synchronize()
        differing = int((bits(out) != bits(a * b + c)).sum())
        self.assertGreater(differing, n // 100,
                           f"a fused multiply-add differed from eager in only {differing} of {n}")

    @unittest.skipUnless(CUDA, NO_CUDA)
    def test_a_group_never_needs_its_own_output(self):
        # `a + b` would join `a`'s kernel, but `b = sin(a)` runs eagerly between
        # them: one kernel for both adds would need `b` before producing `a`.
        class Cycle(torch.nn.Module):
            def forward(self, x, y):
                a = x + y
                b = torch.sin(a)
                return a + b

        x, y = operands(2)
        cm, reports = compile_with_reports(Cycle())
        with torch.no_grad():
            out = cm(x, y)
            ref = Cycle()(x, y)
        self.assertEqual([k.ops for k in reports[-1].kernels], [["add"], ["add"]])
        self.assertTrue(torch.equal(bits(out), bits(ref)))

    @unittest.skipUnless(CUDA, NO_CUDA)
    def test_what_is_not_lowered_runs_eagerly_and_says_why(self):
        class Mixed(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(32, 32)

            def forward(self, x):
                h = self.lin(x)
                return torch.nn.functional.silu(h) + (-h) / 3.0 + h * h + x

        torch.manual_seed(1)
        model = Mixed().cuda()
        x = torch.randn(4, 32, device="cuda")
        cm, reports = compile_with_reports(model)
        with torch.no_grad():
            out = cm(x)
            ref = model(x)
        self.assertTrue(torch.equal(bits(out), bits(ref)))
        report = reports[-1]
        why = {what: reason for _, what, reason in report.not_lowered}
        self.assertIn("negation is not lowered", why["call_function `neg`"])
        self.assertIn("division is not lowered", why["call_function `truediv`"])
        self.assertIn("is not lowered", why["call_function `silu`"])
        self.assertIn("is not lowered", why["call_function `linear`"])
        self.assertGreaterEqual(report.lowered_ops, 3, report.summary())

    @unittest.skipUnless(CUDA, NO_CUDA)
    def test_scalar_operands_are_exact_or_not_lowered(self):
        (x,) = operands(1)
        cases = {
            "x + 0.0": lambda t: t + 0.0, "x - 0.0": lambda t: t - 0.0,
            "x + (-0.0)": lambda t: t + (-0.0), "0.0 - x": lambda t: 0.0 - t,
            "x * -1.0": lambda t: t * -1.0, "x * 0.0": lambda t: t * 0.0,
            "x * 0.1": lambda t: t * 0.1, "x + 3": lambda t: t + 3,
            "2.5 - x": lambda t: 2.5 - t, "x * 1e-40": lambda t: t * 1e-40,
            "x - (-7.25)": lambda t: t - (-7.25), "x * 1.0": lambda t: t * 1.0,
        }

        class One(torch.nn.Module):
            def __init__(self, fn):
                super().__init__()
                self.fn = fn

            def forward(self, t):
                return self.fn(t)

        for name, fn in cases.items():
            cm, reports = compile_with_reports(One(fn))
            with torch.no_grad():
                out = cm(x)
            self.assertTrue(torch.equal(bits(out), bits(fn(x))), f"{name} differs from eager")
            kernels = reports[-1].kernels
            if name == "x * 1.0":
                # ptxas deletes a multiply by 1.0, so a NaN would keep its payload.
                self.assertEqual(kernels, [], "a multiply by 1.0 must not be lowered")
                self.assertIn("multiply by 1.0", reports[-1].not_lowered[0][2])
            else:
                self.assertEqual([k.launches for k in kernels], [1], f"{name} was not lowered")

    def test_a_cpu_graph_is_not_lowered_and_says_why(self):
        x, y, z = operands(3, device="cpu")
        cm, reports = compile_with_reports(Elementwise())
        with torch.no_grad():
            out = cm(x, y, z)
        self.assertTrue(torch.equal(bits(out), bits(Elementwise()(x, y, z))))
        report = reports[-1]
        self.assertEqual(report.kernels, [])
        self.assertTrue(report.not_lowered)
        self.assertTrue(all("only CUDA is lowered" in why for _, _, why in report.not_lowered),
                        report.summary())

    @unittest.skipUnless(CUDA, NO_CUDA)
    def test_the_module_path_lowers_and_falls_back_at_runtime(self):
        # `y_inductor(module, inputs)`, the call the package's examples make.
        x, y, z = (t[: 64 * 64].reshape(64, 64) for t in operands(3))
        compiled = y_inductor(Elementwise(), [x, y, z])
        self.assertEqual([k.ops for k in compiled.y_report.kernels], [["add", "relu", "mul"], ["sub"]])
        with torch.no_grad():
            self.assertTrue(torch.equal(bits(compiled(x, y, z)), bits(Elementwise()(x, y, z))))
            # A non-contiguous input with the compiled shape takes the original
            # subgraph, and the report counts it.
            xt = x.t()
            self.assertFalse(xt.is_contiguous())
            self.assertTrue(torch.equal(bits(compiled(xt, y, z)), bits(Elementwise()(xt, y, z))))
        first = compiled.y_report.kernels[0]
        self.assertEqual(first.launches, 1)
        self.assertEqual(first.fallbacks, {"a non-contiguous input": 1})

    @unittest.skipUnless(CUDA, NO_CUDA)
    def test_broadcasting_and_other_dtypes_are_not_lowered(self):
        class Bias(torch.nn.Module):
            def forward(self, x, b):
                return x + b

        x = torch.randn(4, 8, device="cuda")
        b = torch.randn(8, device="cuda")
        cm, reports = compile_with_reports(Bias())
        with torch.no_grad():
            out = cm(x, b)
        self.assertTrue(torch.equal(bits(out), bits(x + b)))
        self.assertEqual(reports[-1].kernels, [])
        self.assertIn("broadcasting", reports[-1].not_lowered[0][2])

        xd = torch.randn(64, device="cuda", dtype=torch.float64)
        cm, reports = compile_with_reports(Bias())
        with torch.no_grad():
            out = cm(xd, xd)
        self.assertTrue(torch.equal(out, xd + xd))
        self.assertEqual(reports[-1].kernels, [])
        self.assertIn("only float32 is lowered", reports[-1].not_lowered[0][2])


if __name__ == "__main__":
    unittest.main()
