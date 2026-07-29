
typedef unsigned long long u64;

extern "C" __global__ void test_full_gpu(u64* out, const u64* A, const u64* B) {
    u64 a0=A[0], a1=A[1], a2=A[2], a3=A[3];
    u64 b0=B[0], b1=B[1], b2=B[2], b3=B[3];

    u64 k0 = 0xbc1e0a6c0fffffffULL;
    u64 k1 = 0xd7cc17b786468f6eULL;
    u64 k2 = 0x47afba497e7ea7a2ULL;
    u64 k3 = 0xcf9bb18d1ece5fd6ULL;

    u64 p0 = 0x43e1f593f0000001ULL;
    u64 p1 = 0x2833e84879b97091ULL;
    u64 p2 = 0xb85045b68181585dULL;
    u64 p3 = 0x30644e72e131a029ULL;

    u64 c0=0, c1=0, c2=0, c3=0, c4=0, c5=0, c6=0, c7=0;
    u64 tk0=0, tk1=0, tk2=0, tk3=0, tk4=0, tk5=0, tk6=0, tk7=0;
    u64 tlo=0, thi=0;

    // 4x4 Row Multiplication
    asm volatile (
        "mul.lo.u64 %0, %10, %14;\n\t"
        "mul.hi.u64 %1, %10, %14;\n\t"

        "mul.lo.u64 %8, %11, %14;\n\t"
        "mul.hi.u64 %9, %11, %14;\n\t"
        "add.cc.u64 %1, %1, %8;\n\t"
        "addc.u64 %2, %9, 0;\n\t"

        "mul.lo.u64 %8, %12, %14;\n\t"
        "mul.hi.u64 %9, %12, %14;\n\t"
        "add.cc.u64 %2, %2, %8;\n\t"
        "addc.u64 %3, %9, 0;\n\t"

        "mul.lo.u64 %8, %13, %14;\n\t"
        "mul.hi.u64 %9, %13, %14;\n\t"
        "add.cc.u64 %3, %3, %8;\n\t"
        "addc.u64 %4, %9, 0;\n\t"

        "mul.lo.u64 %8, %10, %15;\n\t"
        "mul.hi.u64 %9, %10, %15;\n\t"
        "add.cc.u64 %1, %1, %8;\n\t"
        "addc.cc.u64 %2, %2, %9;\n\t"
        "addc.cc.u64 %3, %3, 0;\n\t"
        "addc.u64 %4, %4, 0;\n\t"

        "mul.lo.u64 %8, %11, %15;\n\t"
        "mul.hi.u64 %9, %11, %15;\n\t"
        "add.cc.u64 %2, %2, %8;\n\t"
        "addc.cc.u64 %3, %3, %9;\n\t"
        "addc.u64 %4, %4, 0;\n\t"

        "mul.lo.u64 %8, %12, %15;\n\t"
        "mul.hi.u64 %9, %12, %15;\n\t"
        "add.cc.u64 %3, %3, %8;\n\t"
        "addc.cc.u64 %4, %4, %9;\n\t"
        "addc.u64 %5, 0, 0;\n\t"

        "mul.lo.u64 %8, %13, %15;\n\t"
        "mul.hi.u64 %9, %13, %15;\n\t"
        "add.cc.u64 %4, %4, %8;\n\t"
        "addc.u64 %5, %5, %9;\n\t"

        "mul.lo.u64 %8, %10, %16;\n\t"
        "mul.hi.u64 %9, %10, %16;\n\t"
        "add.cc.u64 %2, %2, %8;\n\t"
        "addc.cc.u64 %3, %3, %9;\n\t"
        "addc.cc.u64 %4, %4, 0;\n\t"
        "addc.u64 %5, %5, 0;\n\t"

        "mul.lo.u64 %8, %11, %16;\n\t"
        "mul.hi.u64 %9, %11, %16;\n\t"
        "add.cc.u64 %3, %3, %8;\n\t"
        "addc.cc.u64 %4, %4, %9;\n\t"
        "addc.u64 %5, %5, 0;\n\t"

        "mul.lo.u64 %8, %12, %16;\n\t"
        "mul.hi.u64 %9, %12, %16;\n\t"
        "add.cc.u64 %4, %4, %8;\n\t"
        "addc.cc.u64 %5, %5, %9;\n\t"
        "addc.u64 %6, 0, 0;\n\t"

        "mul.lo.u64 %8, %13, %16;\n\t"
        "mul.hi.u64 %9, %13, %16;\n\t"
        "add.cc.u64 %5, %5, %8;\n\t"
        "addc.u64 %6, %6, %9;\n\t"

        "mul.lo.u64 %8, %10, %17;\n\t"
        "mul.hi.u64 %9, %10, %17;\n\t"
        "add.cc.u64 %3, %3, %8;\n\t"
        "addc.cc.u64 %4, %4, %9;\n\t"
        "addc.cc.u64 %5, %5, 0;\n\t"
        "addc.u64 %6, %6, 0;\n\t"

        "mul.lo.u64 %8, %11, %17;\n\t"
        "mul.hi.u64 %9, %11, %17;\n\t"
        "add.cc.u64 %4, %4, %8;\n\t"
        "addc.cc.u64 %5, %5, %9;\n\t"
        "addc.u64 %6, %6, 0;\n\t"

        "mul.lo.u64 %8, %10, %17;\n\t"
        "mul.hi.u64 %9, %10, %17;\n\t"
        "add.cc.u64 %5, %5, %8;\n\t"
        "addc.cc.u64 %6, %6, %9;\n\t"
        "addc.u64 %7, 0, 0;\n\t"

        "mul.lo.u64 %8, %13, %17;\n\t"
        "mul.hi.u64 %9, %13, %17;\n\t"
        "add.cc.u64 %6, %6, %8;\n\t"
        "addc.u64 %7, %7, %9;\n\t"
        : "=&l"(c0), "=&l"(c1), "=&l"(c2), "=&l"(c3), "=&l"(c4), "=&l"(c5), "=&l"(c6), "=&l"(c7),
          "=&l"(tlo), "=&l"(thi)
        : "l"(a0), "l"(a1), "l"(a2), "l"(a3), "l"(b0), "l"(b1), "l"(b2), "l"(b3)
    );

    // K-reduction loop (64 passes)
    for (int pass = 0; pass < 64; pass++) {
        a0 = c4; a1 = c5; a2 = c6; a3 = c7;
        c4 = 0; c5 = 0; c6 = 0; c7 = 0;
        tk0 = 0; tk1 = 0; tk2 = 0; tk3 = 0; tk4 = 0; tk5 = 0; tk6 = 0; tk7 = 0;

        asm volatile (
            "mul.lo.u64 %0, %10, %14;\n\t"
            "mul.hi.u64 %1, %10, %14;\n\t"

            "mul.lo.u64 %8, %11, %14;\n\t"
            "mul.hi.u64 %9, %11, %14;\n\t"
            "add.cc.u64 %1, %1, %8;\n\t"
            "addc.u64 %2, %9, 0;\n\t"

            "mul.lo.u64 %8, %12, %14;\n\t"
            "mul.hi.u64 %9, %12, %14;\n\t"
            "add.cc.u64 %2, %2, %8;\n\t"
            "addc.u64 %3, %9, 0;\n\t"

            "mul.lo.u64 %8, %13, %14;\n\t"
            "mul.hi.u64 %9, %13, %14;\n\t"
            "add.cc.u64 %3, %3, %8;\n\t"
            "addc.u64 %4, %9, 0;\n\t"

            "mul.lo.u64 %8, %10, %15;\n\t"
            "mul.hi.u64 %9, %10, %15;\n\t"
            "add.cc.u64 %1, %1, %8;\n\t"
            "addc.cc.u64 %2, %2, %9;\n\t"
            "addc.cc.u64 %3, %3, 0;\n\t"
            "addc.u64 %4, %4, 0;\n\t"

            "mul.lo.u64 %8, %11, %15;\n\t"
            "mul.hi.u64 %9, %11, %15;\n\t"
            "add.cc.u64 %2, %2, %8;\n\t"
            "addc.cc.u64 %3, %3, %9;\n\t"
            "addc.u64 %4, %4, 0;\n\t"

            "mul.lo.u64 %8, %12, %15;\n\t"
            "mul.hi.u64 %9, %12, %15;\n\t"
            "add.cc.u64 %3, %3, %8;\n\t"
            "addc.cc.u64 %4, %4, %9;\n\t"
            "addc.u64 %5, 0, 0;\n\t"

            "mul.lo.u64 %8, %13, %15;\n\t"
            "mul.hi.u64 %9, %13, %15;\n\t"
            "add.cc.u64 %4, %4, %8;\n\t"
            "addc.u64 %5, %5, %9;\n\t"

            "mul.lo.u64 %8, %10, %16;\n\t"
            "mul.hi.u64 %9, %10, %16;\n\t"
            "add.cc.u64 %2, %2, %8;\n\t"
            "addc.cc.u64 %3, %3, %9;\n\t"
            "addc.cc.u64 %4, %4, 0;\n\t"
            "addc.u64 %5, %5, 0;\n\t"

            "mul.lo.u64 %8, %11, %16;\n\t"
            "mul.hi.u64 %9, %11, %16;\n\t"
            "add.cc.u64 %3, %3, %8;\n\t"
            "addc.cc.u64 %4, %4, %9;\n\t"
            "addc.u64 %5, %5, 0;\n\t"

            "mul.lo.u64 %8, %12, %16;\n\t"
            "mul.hi.u64 %9, %12, %16;\n\t"
            "add.cc.u64 %4, %4, %8;\n\t"
            "addc.cc.u64 %5, %5, %9;\n\t"
            "addc.u64 %6, 0, 0;\n\t"

            "mul.lo.u64 %8, %13, %16;\n\t"
            "mul.hi.u64 %9, %13, %16;\n\t"
            "add.cc.u64 %5, %5, %8;\n\t"
            "addc.u64 %6, %6, %9;\n\t"

            "mul.lo.u64 %8, %10, %17;\n\t"
            "mul.hi.u64 %9, %10, %17;\n\t"
            "add.cc.u64 %3, %3, %8;\n\t"
            "addc.cc.u64 %4, %4, %9;\n\t"
            "addc.cc.u64 %5, %5, 0;\n\t"
            "addc.u64 %6, %6, 0;\n\t"

            "mul.lo.u64 %8, %11, %17;\n\t"
            "mul.hi.u64 %9, %11, %17;\n\t"
            "add.cc.u64 %4, %4, %8;\n\t"
            "addc.cc.u64 %5, %5, %9;\n\t"
            "addc.u64 %6, %6, 0;\n\t"

            "mul.lo.u64 %8, %12, %17;\n\t"
            "mul.hi.u64 %9, %12, %17;\n\t"
            "add.cc.u64 %5, %5, %8;\n\t"
            "addc.cc.u64 %6, %6, %9;\n\t"
            "addc.u64 %7, 0, 0;\n\t"

            "mul.lo.u64 %8, %13, %17;\n\t"
            "mul.hi.u64 %9, %13, %17;\n\t"
            "add.cc.u64 %6, %6, %8;\n\t"
            "addc.u64 %7, %7, %9;\n\t"
            : "=&l"(tk0), "=&l"(tk1), "=&l"(tk2), "=&l"(tk3), "=&l"(tk4), "=&l"(tk5), "=&l"(tk6), "=&l"(tk7),
              "=&l"(tlo), "=&l"(thi)
            : "l"(a0), "l"(a1), "l"(a2), "l"(a3), "l"(k0), "l"(k1), "l"(k2), "l"(k3)
        );

        asm volatile (
            "add.cc.u64 %0, %0, %8;\n\t"
            "addc.cc.u64 %1, %1, %9;\n\t"
            "addc.cc.u64 %2, %2, %10;\n\t"
            "addc.cc.u64 %3, %3, %11;\n\t"
            "addc.cc.u64 %4, %12, 0;\n\t"
            "addc.cc.u64 %5, %13, 0;\n\t"
            "addc.cc.u64 %6, %14, 0;\n\t"
            "addc.u64 %7, %15, 0;\n\t"
            : "+l"(c0), "+l"(c1), "+l"(c2), "+l"(c3), "+l"(c4), "+l"(c5), "+l"(c6), "+l"(c7)
            : "l"(tk0), "l"(tk1), "l"(tk2), "l"(tk3), "l"(tk4), "l"(tk5), "l"(tk6), "l"(tk7)
        );
    }

    // 8 Conditional Subtractions mod p
    for (int step = 0; step < 8; step++) {
        u64 r0, r1, r2, r3, r4;
        asm volatile (
            "sub.cc.u64 %0, %5, %10;\n\t"
            "subc.cc.u64 %1, %6, %11;\n\t"
            "subc.cc.u64 %2, %7, %12;\n\t"
            "subc.cc.u64 %3, %8, %13;\n\t"
            "subc.u64 %4, %9, 0;\n\t"
            "setp.eq.u64 %%p1, %4, 0;\n\t"
            "selp.b64 %5, %0, %5, %%p1;\n\t"
            "selp.b64 %6, %1, %6, %%p1;\n\t"
            "selp.b64 %7, %2, %7, %%p1;\n\t"
            "selp.b64 %8, %3, %8, %%p1;\n\t"
            "selp.b64 %9, 0, %9, %%p1;\n\t"
            : "=&l"(r0), "=&l"(r1), "=&l"(r2), "=&l"(r3), "=&l"(r4),
              "+l"(c0), "+l"(c1), "+l"(c2), "+l"(c3), "+l"(c4)
            : "l"(p0), "l"(p1), "l"(p2), "l"(p3)
        );
    }

    out[0]=c0; out[1]=c1; out[2]=c2; out[3]=c3;
}
