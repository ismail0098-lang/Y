"""
Unit Test Suite for Y Language Triton Parity Features.
Verifies dynamic autotuning (@autotune, Config, @heuristics), block primitives (cdiv, arange, load, store, dot, where),
and PyTorch Inductor compiler backend (y_inductor).
"""

import unittest
import torch
import torch.nn as nn
import sys
import os

# Ensure package is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import y_lang
from y_lang import autotune, Config, heuristics, y_inductor
from y_lang.ops.block import cdiv, arange, load, store, dot, where, expand_dims

class TestTritonParity(unittest.TestCase):

    def test_cdiv(self):
        self.assertEqual(cdiv(10, 3), 4)
        self.assertEqual(cdiv(100, 32), 4)
        self.assertEqual(cdiv(128, 64), 2)
        t_a = torch.tensor(10)
        t_b = torch.tensor(3)
        self.assertEqual(cdiv(t_a, t_b).item(), 4)

    def test_arange(self):
        t = arange(0, 16, dtype=torch.int32, device="cpu")
        self.assertEqual(t.shape[0], 16)
        self.assertEqual(t[0].item(), 0)
        self.assertEqual(t[15].item(), 15)

    def test_expand_dims_and_where(self):
        x = torch.ones((4, 4))
        expanded = expand_dims(x, 0)
        self.assertEqual(expanded.shape, (1, 4, 4))

        cond = torch.tensor([True, False, True, False])
        a = torch.tensor([1, 2, 3, 4])
        b = torch.tensor([10, 20, 30, 40])
        res = where(cond, a, b)
        self.assertTrue(torch.equal(res, torch.tensor([1, 20, 3, 40])))

    def test_masked_load_store(self):
        data = torch.tensor([1.0, 2.0, 3.0, 4.0])
        mask = torch.tensor([True, True, False, False])
        loaded = load(data, mask=mask, other=0.0)
        self.assertTrue(torch.equal(loaded, torch.tensor([1.0, 2.0, 0.0, 0.0])))

        target = torch.zeros(4)
        val = torch.tensor([9.0, 8.0, 7.0, 6.0])
        store(target, val, mask=mask)
        self.assertTrue(torch.equal(target, torch.tensor([9.0, 8.0, 0.0, 0.0])))

    def test_block_dot(self):
        a = torch.randn(16, 32)
        b = torch.randn(32, 16)
        res = dot(a, b)
        self.assertEqual(res.shape, (16, 16))
        expected = torch.matmul(a, b)
        self.assertTrue(torch.allclose(res, expected, atol=1e-4))

    def test_autotune_decorator(self):
        configs = [
            Config(cta_m=64, cta_n=64, cta_k=32, num_warps=4),
            Config(cta_m=128, cta_n=128, cta_k=32, num_warps=8),
        ]

        @autotune(configs=configs, key=["M", "N"])
        def mock_kernel(config, x, M=1024, N=1024):
            return x * config.cta_m

        x = torch.ones(10)
        out = mock_kernel(x, M=1024, N=1024)
        self.assertIsNotNone(out)

    def test_heuristics_decorator(self):
        @heuristics({"BLOCK_SIZE": lambda x: 128 if x.shape[0] > 100 else 64})
        def mock_kernel(x, BLOCK_SIZE=None):
            return BLOCK_SIZE

        x_large = torch.zeros(200)
        self.assertEqual(mock_kernel(x_large), 128)

        x_small = torch.zeros(50)
        self.assertEqual(mock_kernel(x_small), 64)

    def test_y_inductor_compiler(self):
        class SimpleMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(16, 16)
                self.act = nn.ReLU()

            def forward(self, x):
                return self.act(self.fc(x))

        model = SimpleMLP()
        x = torch.randn(4, 16)
        
        # Test direct FX GraphModule compilation via y_inductor
        gm = torch.fx.symbolic_trace(model)
        compiled_fn = y_inductor(gm, [x])
        out = compiled_fn(x)

        expected = model(x)
        self.assertTrue(torch.allclose(out, expected, atol=1e-4))
        self.assertTrue(getattr(compiled_fn, "_y_compiled", False))
        self.assertEqual(getattr(compiled_fn, "_backend", None), "y_inductor")

if __name__ == "__main__":
    unittest.main()

