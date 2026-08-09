// tests/y_runtime_shim.c
// Implementation of block_ptr2d_load and block_ptr2d_store runtime helpers for Y compiler LLVM IR backend.

#include <stdint.h>

int32_t block_ptr2d_load(uintptr_t ptr_val, int row, int col, int stride, int max_r, int max_c) {
    const float* ptr = (const float*)ptr_val;
    float val = ptr[row * stride + col];
    // Return float bit pattern reinterpreted as int32
    int32_t res;
    *(float*)&res = val;
    return res;
}

int32_t block_ptr2d_store(uintptr_t ptr_val, int row, int col, int stride, int max_r, int max_c, int32_t val_bits) {
    float* ptr = (float*)ptr_val;
    float val;
    *(int32_t*)&val = val_bits;
    ptr[row * stride + col] = val;
    return 0;
}
