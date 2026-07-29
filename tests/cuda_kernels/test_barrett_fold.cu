
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

__device__ void add256(u64* R, const u64* A, const u64* B) {
    u64 carry = 0;
    for (int i = 0; i < 4; i++) {
        u128 sum = (u128)A[i] + (u128)B[i] + (u128)carry;
        R[i] = (u64)sum;
        carry = (u64)(sum >> 64);
    }
}

extern "C" __global__ void test_barrett_fold(u64* out, const u64* x, const u64* y) {
    u64 C[8];
    mul4x4(C, x, y);

    u64 P[4] = {0x43e1f593f0000001ULL, 0x2833e84879b97091ULL, 0xb85045b68181585dULL, 0x30644e72e131a029ULL};
    u64 K[4] = {0xbc1e0a6c0fffffffULL, 0xd7cc17b786468f6eULL, 0x47afba497e7ea7a2ULL, 0xcf9bb18d1ece5fd6ULL};
    u64 MU[4] = {0x620703a6be1de925ULL, 0x144852009e880ae6ULL, 0xab074a5868073014ULL, 0x54a47462623a04a7ULL};

    // q_approx = (C_hi * MU) >> 256
    u64 Q_prod[8];
    mul4x4(Q_prod, &C[4], MU);
    u64* Q = &Q_prod[4];

    // QP = Q * P
    u64 QP[8];
    mul4x4(QP, Q, P);

    // Compute difference R512 = C - QP (512-bit)
    u64 R512[8];
    u64 borrow = 0;
    for (int i = 0; i < 8; i++) {
        u128 diff = (u128)C[i] - (u128)QP[i] - (u128)borrow;
        R512[i] = (u64)diff;
        borrow = (diff >> 127) & 1;
    }

    // Fold high limbs R512[4..7] using 2^256 = P + K => R_folded = R_lo + R_hi * K
    u64 RK[8];
    mul4x4(RK, &R512[4], K);

    u64 R[4];
    u64 carry = 0;
    for (int i = 0; i < 4; i++) {
        u128 sum = (u128)R512[i] + (u128)RK[i] + (u128)carry;
        R[i] = (u64)sum;
        carry = (u64)(sum >> 64);
    }

    // Handle underflow if borrow occurred
    if (borrow) {
        u64 tmp[4];
        add256(tmp, R, P);
        for (int i = 0; i < 4; i++) R[i] = tmp[i];
    }

    // Reduce R >= P (at most 4 subtractions)
    for (int step = 0; step < 4; step++) {
        if (ge256(R, P)) {
            u64 tmp[4];
            sub256(tmp, R, P);
            for (int i = 0; i < 4; i++) R[i] = tmp[i];
        }
    }

    for (int i = 0; i < 4; i++) out[i] = R[i];
}
