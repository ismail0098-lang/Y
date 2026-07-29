
typedef unsigned long long u64;

__device__ void mul4x4(u64* C, const u64* A, const u64* B) {
    #pragma unroll
    for (int i = 0; i < 8; i++) C[i] = 0;

    for (int i = 0; i < 4; i++) {
        u64 carry = 0;
        for (int j = 0; j < 4; j++) {
            u64 p_lo = A[i] * B[j];
            u64 p_hi = __umul64hi(A[i], B[j]);

            u64 s1 = C[i + j] + p_lo;
            u64 c1 = (s1 < C[i + j]);
            C[i + j] = s1;

            u64 s2 = C[i + j + 1] + p_hi + carry + c1;
            u64 c2 = (s2 < C[i + j + 1]) || (carry + c1 > s2); // check overflow
            C[i + j + 1] = s2;

            carry = c2;
        }
    }
}

extern "C" __global__ void test_cuda_c_mul(u64* out, const u64* x, const u64* y) {
    u64 C[8];
    mul4x4(C, x, y);

    u64 K[4] = {0xbc1e0a6c0fffffffULL, 0xd7cc17b786468f6eULL, 0x47afba497e7ea7a2ULL, 0xcf9bb18d1ece5fd6ULL};
    u64 P[4] = {0x43e1f593f0000001ULL, 0x2833e84879b97091ULL, 0xb85045b68181585dULL, 0x30644e72e131a029ULL};

    // Pass 1: T1 = C_lo + C_hi * K
    u64 CK1[8];
    mul4x4(CK1, &C[4], K);

    u64 T1[8];
    u64 carry = 0;
    for (int i = 0; i < 4; i++) {
        u64 s = C[i] + CK1[i] + carry;
        carry = (s < C[i]) || (s < CK1[i]);
        T1[i] = s;
    }
    for (int i = 4; i < 8; i++) {
        u64 s = CK1[i] + carry;
        carry = (s < CK1[i]);
        T1[i] = s;
    }

    // Pass 2: T2 = T1_lo + T1_hi * K
    u64 CK2[8];
    mul4x4(CK2, &T1[4], K);

    u64 T2[8];
    carry = 0;
    for (int i = 0; i < 4; i++) {
        u64 s = T1[i] + CK2[i] + carry;
        carry = (s < T1[i]) || (s < CK2[i]);
        T2[i] = s;
    }
    for (int i = 4; i < 8; i++) {
        u64 s = CK2[i] + carry;
        carry = (s < CK2[i]);
        T2[i] = s;
    }

    // Pass 3: T3 = T2_lo + T2_hi * K
    u64 CK3[8];
    mul4x4(CK3, &T2[4], K);

    u64 T3[8];
    carry = 0;
    for (int i = 0; i < 4; i++) {
        u64 s = T2[i] + CK3[i] + carry;
        carry = (s < T2[i]) || (s < CK3[i]);
        T3[i] = s;
    }
    for (int i = 4; i < 8; i++) {
        u64 s = CK3[i] + carry;
        carry = (s < CK3[i]);
        T3[i] = s;
    }

    // Reduction mod P
    for (int step = 0; step < 4; step++) {
        u64 R[4];
        u64 borrow = 0;
        for (int i = 0; i < 4; i++) {
            u64 diff = T3[i] - P[i] - borrow;
            borrow = (T3[i] < P[i] + borrow);
            R[i] = diff;
        }
        if (!borrow && T3[4] == 0) {
            for (int i = 0; i < 4; i++) T3[i] = R[i];
        }
    }

    for (int i = 0; i < 4; i++) out[i] = T3[i];
}
