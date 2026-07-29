
typedef unsigned long long u64;

extern "C" __global__ void test_cond_sub(u64* out, const u64* C_in) {
    u64 c0 = C_in[0], c1 = C_in[1], c2 = C_in[2], c3 = C_in[3], c4 = C_in[4];

    u64 p0 = 0x43e1f593f0000001ULL;
    u64 p1 = 0x2833e84879b97091ULL;
    u64 p2 = 0xb85045b68181585dULL;
    u64 p3 = 0x30644e72e131a029ULL;

    for (int step = 0; step < 8; step++) {
        u64 r0, r1, r2, r3, r4;

        asm volatile (
            ".reg .pred %%pred_sub;\n\t"
            "sub.cc.u64 %0, %5, %10;\n\t"
            "subc.cc.u64 %1, %6, %11;\n\t"
            "subc.cc.u64 %2, %7, %12;\n\t"
            "subc.cc.u64 %3, %8, %13;\n\t"
            "subc.u64 %4, %9, 0;\n\t"
            "setp.eq.u64 %%pred_sub, %4, 0;\n\t"
            "selp.b64 %5, %0, %5, %%pred_sub;\n\t"
            "selp.b64 %6, %1, %6, %%pred_sub;\n\t"
            "selp.b64 %7, %2, %7, %%pred_sub;\n\t"
            "selp.b64 %8, %3, %8, %%pred_sub;\n\t"
            "selp.b64 %9, 0, %9, %%pred_sub;\n\t"
            : "=&l"(r0), "=&l"(r1), "=&l"(r2), "=&l"(r3), "=&l"(r4),
              "+l"(c0), "+l"(c1), "+l"(c2), "+l"(c3), "+l"(c4)
            : "l"(p0), "l"(p1), "l"(p2), "l"(p3)
        );
    }

    out[0] = c0; out[1] = c1; out[2] = c2; out[3] = c3; out[4] = c4;
}
