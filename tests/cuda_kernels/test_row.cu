
typedef unsigned long long u64;

extern "C" __global__ void test_row_mul(u64* out, const u64* A, const u64* B) {
    u64 a0=A[0], a1=A[1], a2=A[2], a3=A[3];
    u64 b0=B[0], b1=B[1], b2=B[2], b3=B[3];
    u64 c0=0, c1=0, c2=0, c3=0, c4=0, c5=0, c6=0, c7=0;

    u64 tlo=0, thi=0;

    asm volatile (
        // Row 0 (a0..a3 * b0):
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

        // Row 1 (a0..a3 * b1):
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

        // Row 2 (a0..a3 * b2):
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

        // Row 3 (a0..a3 * b3):
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

        : "=&l"(c0), "=&l"(c1), "=&l"(c2), "=&l"(c3), "=&l"(c4), "=&l"(c5), "=&l"(c6), "=&l"(c7),
          "=&l"(tlo), "=&l"(thi)
        : "l"(a0), "l"(a1), "l"(a2), "l"(a3), "l"(b0), "l"(b1), "l"(b2), "l"(b3)
    );

    out[0]=c0; out[1]=c1; out[2]=c2; out[3]=c3; out[4]=c4; out[5]=c5; out[6]=c6; out[7]=c7;
}
