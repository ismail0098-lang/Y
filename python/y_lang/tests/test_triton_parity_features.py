"""
Unit test suite for Triton Parity Features #2, #3, #5, #6 in Y Language python package.
"""

import unittest
import torch
import y_lang
from y_lang.inductor import y_inductor
from y_lang.interpreter import CPUInterpreter, debug_interpret, CPUInterpreterError
from y_lang.ops import (
    block_scan,
    block_sort,
    block_where,
    associative_reduce,
    fp8_gemm,
    int4_weight_only_gemm,
    paged_attention,
)


def test_feature_2_pytorch_inductor_backend():
    """Tests feature #2: PyTorch Inductor Backend Integration."""
    class SimpleModel(torch.nn.Module):
        def forward(self, x, y):
            return torch.relu(x + y)

    model = SimpleModel()
    x = torch.randn(16, 32)
    y = torch.randn(16, 32)

    compiled_model = y_inductor(model, [x, y])
    res = compiled_model(x, y)
    expected = torch.relu(x + y)

    assert torch.equal(res, expected)
    # CPU tensors: y_inductor lowers elementwise float32 CUDA subgraphs only,
    # so nothing is lowered and the report says why (see test_inductor.py).
    # This used to assert `_y_compiled` was True - a flag the old backend set
    # while compiling nothing at all.
    assert compiled_model.y_report.kernels == []
    assert all("only CUDA is lowered" in why for _, _, why in compiled_model.y_report.not_lowered)

def test_feature_3_block_primitives():
    """Tests feature #3: Block Primitives (scan, sort, where, associative_reduce)."""
    x = torch.tensor([[3.0, 1.0, 4.0], [1.0, 5.0, 9.0]])

    # 1. block_scan
    scan_res = block_scan(x, dim=-1, op="sum")
    expected_scan = torch.cumsum(x, dim=-1)
    assert torch.allclose(scan_res, expected_scan)

    # 2. block_sort
    sort_res = block_sort(x, dim=-1)
    expected_sort = torch.tensor([[1.0, 3.0, 4.0], [1.0, 5.0, 9.0]])
    assert torch.allclose(sort_res, expected_sort)

    # 3. block_where
    cond = x > 2.0
    where_res = block_where(cond, x, torch.zeros_like(x))
    expected_where = torch.where(cond, x, torch.zeros_like(x))
    assert torch.allclose(where_res, expected_where)

    # 4. associative_reduce
    reduce_res = associative_reduce(x, dim=-1, op="sum")
    expected_reduce = torch.sum(x, dim=-1, keepdim=True)
    assert torch.allclose(reduce_res, expected_reduce)

def test_feature_5_cpu_interpreter_debugger():
    """Tests feature #5: CPU Interpreter & Debugger."""
    interp = CPUInterpreter("test_kernel")
    t = torch.randn(4, 8)

    # Valid bounds check
    interp.validate_tensor_bounds(t, (2, 5), "t")
    assert interp.bounds_checks_passed == 1

    # Out of bounds check
    caught = False
    try:
        interp.validate_tensor_bounds(t, (4, 5), "t")
    except CPUInterpreterError:
        caught = True
    assert caught, "Expected CPUInterpreterError was not raised"
    assert interp.bounds_checks_failed == 1


    # Run interpreter loop
    res = debug_interpret("dummy", (1, 1, 1), (256, 1, 1), [t])
    assert res["status"] == "SUCCESS"

def test_feature_6_quantized_and_llm_ops():
    """Tests feature #6: Quantized Operators (FP8 GEMM, INT4 GEMM, Paged Attention)."""
    # 1. FP8 GEMM
    a = torch.randn(8, 16, dtype=torch.float16)
    b = torch.randn(16, 8, dtype=torch.float16)
    fp8_res = fp8_gemm(a, b, scale_a=1.0, scale_b=1.0)
    expected_fp8 = torch.matmul(a, b)
    assert torch.allclose(fp8_res, expected_fp8, atol=1e-2)

    # 2. INT4 Weight Only GEMM
    x = torch.randn(4, 16, dtype=torch.float16)
    w_q4 = torch.randint(0, 15, (16, 8), dtype=torch.uint8)
    scales = torch.ones(1, 8, dtype=torch.float16) * 0.1
    int4_res = int4_weight_only_gemm(x, w_q4, scales)
    assert int4_res.shape == (4, 8)

    # 3. Paged Attention
    num_seqs = 2
    num_heads = 4
    head_dim = 16
    block_size = 16
    max_blocks = 4

    query = torch.randn(num_seqs, num_heads, head_dim)
    key_cache = torch.randn(8, num_heads, head_dim, block_size)
    value_cache = torch.randn(8, num_heads, head_dim, block_size)
    block_tables = torch.tensor([[0, 1, -1, -1], [2, 3, -1, -1]], dtype=torch.int32)
    seq_lens = torch.tensor([16, 24], dtype=torch.int32)

    paged_res = paged_attention(query, key_cache, value_cache, block_tables, seq_lens)
    assert paged_res.shape == (num_seqs, num_heads, head_dim)

if __name__ == "__main__":
    test_feature_2_pytorch_inductor_backend()
    test_feature_3_block_primitives()
    test_feature_5_cpu_interpreter_debugger()
    test_feature_6_quantized_and_llm_ops()
    print("ALL TRITON PARITY FEATURE TESTS PASSED SUCCESSFULLY!")
