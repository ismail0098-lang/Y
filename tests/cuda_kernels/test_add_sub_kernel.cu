
typedef unsigned long long u64;

extern "C" __global__ void test_add_sub_kernel(u64* out_add, u64* out_sub, const u64* A, const u64* B) {
    u64 p0 = 0x43e1f593f0000001ULL;
    u64 p1 = 0x2833e84879b97091ULL;
    u64 p2 = 0xb85045b68181585dULL;
    u64 p3 = 0x30644e72e131a029ULL;

    u64 sa_0 = A[0], sa_1 = A[1], sa_2 = A[2], sa_3 = A[3];
    u64 sb_0 = B[0], sb_1 = B[1], sb_2 = B[2], sb_3 = B[3];

    u64 s_add_0, s_add_1, s_add_2, s_add_3;
    u64 s_sub_0, s_sub_1, s_sub_2, s_sub_3;

    // 256-bit Modular Addition (ptx_emitter logic)
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
            : "l"(sa_0), "l"(sa_1), "l"(sa_2), "l"(sa_3), "l"(sb_0), "l"(sb_1), "l"(sb_2), "l"(sb_3)
        );

        asm volatile (
            "sub.cc.u64 %0, %5, %10;\n\t"
            "subc.cc.u64 %1, %6, %11;\n\t"
            "subc.cc.u64 %2, %7, %12;\n\t"
            "subc.cc.u64 %3, %8, %13;\n\t"
            "subc.u64 %4, %9, 0;\n\t"
            : "=l"(r0), "=l"(r1), "=l"(r2), "=l"(r3), "=l"(r4)
            : "l"(t0), "l"(t1), "l"(t2), "l"(t3), "l"(t4), "l"(p0), "l"(p1), "l"(p2), "l"(p3)
        );

        bool p_sub = (r4 == 0);
        s_add_0 = p_sub ? r0 : t0;
        s_add_1 = p_sub ? r1 : t1;
        s_add_2 = p_sub ? r2 : t2;
        s_add_3 = p_sub ? r3 : t3;
    }

    // 256-bit Modular Subtraction (ptx_emitter logic)
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
            : "l"(sa_0), "l"(sa_1), "l"(sa_2), "l"(sa_3), "l"(sb_0), "l"(sb_1), "l"(sb_2), "l"(sb_3)
        );

        asm volatile (
            "add.cc.u64 %0, %4, %8;\n\t"
            "addc.cc.u64 %1, %5, %9;\n\t"
            "addc.cc.u64 %2, %6, %10;\n\t"
            "addc.u64 %3, %7, %11;\n\t"
            : "=l"(r0), "=l"(r1), "=l"(r2), "=l"(r3)
            : "l"(t0), "l"(t1), "l"(t2), "l"(t3), "l"(p0), "l"(p1), "l"(p2), "l"(p3)
        );

        bool p_borrow = (t4 != 0);
        s_sub_0 = p_borrow ? r0 : t0;
        s_sub_1 = p_borrow ? r1 : t1;
        s_sub_2 = p_borrow ? r2 : t2;
        s_sub_3 = p_borrow ? r3 : t3;
    }

    out_add[0] = s_add_0; out_add[1] = s_add_1; out_add[2] = s_add_2; out_add[3] = s_add_3;
    out_sub[0] = s_sub_0; out_sub[1] = s_sub_1; out_sub[2] = s_sub_2; out_sub[3] = s_sub_3;
}
