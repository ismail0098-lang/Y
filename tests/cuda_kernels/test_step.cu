
typedef unsigned long long u64;

extern "C" __global__ void test_step_kernel(u64* out, const u64* A, const u64* B, const u64* C_in, const u64* carry_in) {
    u64 a = A[0];
    u64 b = B[0];
    u64 c = C_in[0];
    u64 carry = carry_in[0];

    u64 c_out, carry_out;
    u64 tlo, thi;

    asm volatile (
        "mul.lo.u64 %0, %4, %5;\n\t"
        "mul.hi.u64 %1, %4, %5;\n\t"
        "add.cc.u64 %0, %0, %6;\n\t"
        "addc.u64 %1, %1, 0;\n\t"
        "add.cc.u64 %2, %0, %7;\n\t"
        "addc.u64 %3, %1, 0;\n\t"
        : "=&l"(tlo), "=&l"(thi), "=l"(c_out), "=l"(carry_out)
        : "l"(a), "l"(b), "l"(c), "l"(carry)
    );

    out[0] = c_out;
    out[1] = carry_out;
}
