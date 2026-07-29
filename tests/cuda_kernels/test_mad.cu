
typedef unsigned long long u64;

extern "C" __global__ void test_mad_kernel(u64* out, const u64* A, const u64* B) {
    u64 C[8] = {0};

    u64 a0=A[0], a1=A[1], a2=A[2], a3=A[3];
    u64 b0=B[0], b1=B[1], b2=B[2], b3=B[3];

    // Compute C = A * B using PTX inline asm
    asm volatile (
        // i = 0
        "mul.lo.u64 %0, %8, %12;\n\t"
        "mul.hi.u64 %1, %8, %12;\n\t"

        "mad.lo.u64 %1, %8, %13, %1;\n\t"
        "madc.hi.u64 %2, %8, %13, 0;\n\t"

        "mad.lo.u64 %2, %8, %14, %2;\n\t"
        "madc.hi.u64 %3, %8, %14, 0;\n\t"

        "mad.lo.u64 %3, %8, %15, %3;\n\t"
        "madc.hi.u64 %4, %8, %15, 0;\n\t"

        // i = 1
        "mad.lo.u64 %1, %9, %12, %1;\n\t"
        "madc.hi.u64 %2, %9, %12, %2;\n\t"

        "mad.lo.u64 %2, %9, %13, %2;\n\t"
        "madc.hi.u64 %3, %9, %13, %3;\n\t"

        "mad.lo.u64 %3, %9, %14, %3;\n\t"
        "madc.hi.u64 %4, %9, %14, %4;\n\t"

        "mad.lo.u64 %4, %9, %15, %4;\n\t"
        "madc.hi.u64 %5, %9, %15, 0;\n\t"

        // i = 2
        "mad.lo.u64 %2, %10, %12, %2;\n\t"
        "madc.hi.u64 %3, %10, %12, %3;\n\t"

        "mad.lo.u64 %3, %10, %13, %3;\n\t"
        "madc.hi.u64 %4, %10, %13, %4;\n\t"

        "mad.lo.u64 %4, %10, %14, %4;\n\t"
        "madc.hi.u64 %5, %10, %14, %5;\n\t"

        "mad.lo.u64 %5, %10, %15, %5;\n\t"
        "madc.hi.u64 %6, %10, %15, 0;\n\t"

        // i = 3
        "mad.lo.u64 %3, %11, %12, %3;\n\t"
        "madc.hi.u64 %4, %11, %12, %4;\n\t"

        "mad.lo.u64 %4, %11, %13, %4;\n\t"
        "madc.hi.u64 %5, %11, %13, %5;\n\t"

        "mad.lo.u64 %5, %11, %14, %5;\n\t"
        "madc.hi.u64 %6, %11, %14, %6;\n\t"

        "mad.lo.u64 %6, %11, %15, %6;\n\t"
        "madc.hi.u64 %7, %11, %15, 0;\n\t"

        : "=l"(C[0]), "=l"(C[1]), "=l"(C[2]), "=l"(C[3]), "=l"(C[4]), "=l"(C[5]), "=l"(C[6]), "=l"(C[7])
        : "l"(a0), "l"(a1), "l"(a2), "l"(a3), "l"(b0), "l"(b1), "l"(b2), "l"(b3)
    );

    for (int k = 0; k < 8; k++) out[k] = C[k];
}
