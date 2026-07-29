
typedef unsigned long long u64;

extern "C" __global__ void test_add_sub(u64* out, const u64* A, const u64* B) {
    u64 P[4] = {0x43e1f593f0000001ULL, 0x2833e84879b97091ULL, 0xb85045b68181585dULL, 0x30644e72e131a029ULL};

    u64 a0 = A[0], a1 = A[1], a2 = A[2], a3 = A[3];
    u64 b0 = B[0], b1 = B[1], b2 = B[2], b3 = B[3];

    // Addition (a + b) mod p
    u64 add_res[4];
    {
        u64 t0, t1, t2, t3, t4;
        u64 r0, r1, r2, r3, r4;

        asm volatile (
            "add.cc.u64 %0, %5, %9;\n\t"
            "addc.cc.u64 %1, %6, %10;\n\t"
            "addc.cc.u64 %2, %7, %11;\n\t"
            "addc.cc.u64 %3, %8, %12;\n\t"
            "addc.u64 %4, 0, 0;\n\t"
            : "=l"(t0), "=l"(t1), "=l"(t2), "=l"(t3), "=l"(t4)
            : "l"(a0), "l"(a1), "l"(a2), "l"(a3), "l"(b0), "l"(b1), "l"(b2), "l"(b3)
        );

        asm volatile (
            "sub.cc.u64 %0, %5, %10;\n\t"
            "subc.cc.u64 %1, %6, %11;\n\t"
            "subc.cc.u64 %2, %7, %12;\n\t"
            "subc.cc.u64 %3, %8, %13;\n\t"
            "subc.u64 %4, %9, 0;\n\t"
            : "=l"(r0), "=l"(r1), "=l"(r2), "=l"(r3), "=l"(r4)
            : "l"(t0), "l"(t1), "l"(t2), "l"(t3), "l"(t4), "l"(P[0]), "l"(P[1]), "l"(P[2]), "l"(P[3])
        );

        bool p_sub = (r4 == 0);
        add_res[0] = p_sub ? r0 : t0;
        add_res[1] = p_sub ? r1 : t1;
        add_res[2] = p_sub ? r2 : t2;
        add_res[3] = p_sub ? r3 : t3;
    }

    // Subtraction (a - b) mod p
    u64 sub_res[4];
    {
        u64 t0, t1, t2, t3, t4;
        u64 r0, r1, r2, r3;

        asm volatile (
            "sub.cc.u64 %0, %5, %9;\n\t"
            "subc.cc.u64 %1, %6, %10;\n\t"
            "subc.cc.u64 %2, %7, %11;\n\t"
            "subc.cc.u64 %3, %8, %12;\n\t"
            "subc.u64 %4, 0, 0;\n\t"
            : "=l"(t0), "=l"(t1), "=l"(t2), "=l"(t3), "=l"(t4)
            : "l"(a0), "l"(a1), "l"(a2), "l"(a3), "l"(b0), "l"(b1), "l"(b2), "l"(b3)
        );

        asm volatile (
            "add.cc.u64 %0, %4, %8;\n\t"
            "addc.cc.u64 %1, %5, %9;\n\t"
            "addc.cc.u64 %2, %6, %10;\n\t"
            "addc.u64 %3, %7, %11;\n\t"
            : "=l"(r0), "=l"(r1), "=l"(r2), "=l"(r3)
            : "l"(t0), "l"(t1), "l"(t2), "l"(t3), "l"(P[0]), "l"(P[1]), "l"(P[2]), "l"(P[3])
        );

        bool p_borrow = (t4 != 0);
        sub_res[0] = p_borrow ? r0 : t0;
        sub_res[1] = p_borrow ? r1 : t1;
        sub_res[2] = p_borrow ? r2 : t2;
        sub_res[3] = p_borrow ? r3 : t3;
    }

    out[0] = add_res[0]; out[1] = add_res[1]; out[2] = add_res[2]; out[3] = add_res[3];
    out[4] = sub_res[0]; out[5] = sub_res[1]; out[6] = sub_res[2]; out[7] = sub_res[3];
}
