
typedef unsigned long long u64;
typedef unsigned __int128 u128;

__device__ void mul4x4(u64* C, const u64* A, const u64* B) {
    for (int i = 0; i < 8; i++) C[i] = 0;

    for (int i = 0; i < 4; i++) {
        u64 carry = 0;
        for (int j = 0; j < 4; j++) {
            u128 prod = (u128)A[i] * (u128)B[j] + (u128)C[i + j] + (u128)carry;
            C[i + j] = (u64)prod;
            carry = (u64)(prod >> 64);
        }
        C[i + 4] += carry;
    }
}

__device__ bool ge256(const u64* A, const u64* B) {
    for (int i = 3; i >= 0; i--) {
        if (A[i] > B[i]) return true;
        if (A[i] < B[i]) return false;
    }
    return true;
}

__device__ void sub256(u64* R, const u64* A, const u64* B) {
    u64 borrow = 0;
    for (int i = 0; i < 4; i++) {
        u128 diff = (u128)A[i] - (u128)B[i] - (u128)borrow;
        R[i] = (u64)diff;
        borrow = (diff >> 127) & 1;
    }
}

extern "C" __global__ void test_barrett_mul(u64* out, const u64* x, const u64* y) {
    u64 C[8];
    mul4x4(C, x, y);

    u64 P[4] = {0x43e1f593f0000001ULL, 0x2833e84879b97091ULL, 0xb85045b68181585dULL, 0x30644e72e131a029ULL};
    u64 MU[4] = {0x620703a6be1de925ULL, 0x144852009e880ae6ULL, 0xab074a5868073014ULL, 0x54a47462623a04a7ULL};

    // 1. q_full = C * MU (512-bit x 256-bit -> taking high 256 bits above bit 512)
    // C_hi = &C[4]
    u64 Q_prod[8];
    mul4x4(Q_prod, &C[4], MU);
    u64* Q = &Q_prod[4];

    // 2. QP = Q * P (256-bit x 256-bit -> 512-bit)
    u64 QP[8];
    mul4x4(QP, Q, P);

    // 3. R = C - QP
    u64 R[8];
    u64 borrow = 0;
    for (int i = 0; i < 8; i++) {
        u128 diff = (u128)C[i] - (u128)QP[i] - (u128)borrow;
        R[i] = (u64)diff;
        borrow = (diff >> 127) & 1;
    }

    // 4. Conditional subtraction of P while R >= P
    for (int step = 0; step < 4; step++) {
        if (ge256(R, P)) {
            u64 sub[4];
            sub256(sub, R, P);
            for (int i = 0; i < 4; i++) R[i] = sub[i];
        }
    }

    for (int i = 0; i < 4; i++) out[i] = R[i];
}
