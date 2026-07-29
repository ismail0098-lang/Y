
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

extern "C" __global__ void test_cuda_c_mul(u64* out, const u64* x, const u64* y) {
    u64 C[8];
    mul4x4(C, x, y);

    u64 K[4] = {0xbc1e0a6c0fffffffULL, 0xd7cc17b786468f6eULL, 0x47afba497e7ea7a2ULL, 0xcf9bb18d1ece5fd6ULL};
    u64 P[4] = {0x43e1f593f0000001ULL, 0x2833e84879b97091ULL, 0xb85045b68181585dULL, 0x30644e72e131a029ULL};

    u64 T[8];
    for (int i = 0; i < 8; i++) T[i] = C[i];

    while (T[4] > 0 || T[5] > 0 || T[6] > 0 || T[7] > 0) {
        u64 CK[8];
        mul4x4(CK, &T[4], K);

        u64 next_T[8];
        u64 carry = 0;
        for (int i = 0; i < 4; i++) {
            u128 sum = (u128)T[i] + (u128)CK[i] + (u128)carry;
            next_T[i] = (u64)sum;
            carry = (u64)(sum >> 64);
        }
        for (int i = 4; i < 8; i++) {
            u128 sum = (u128)CK[i] + (u128)carry;
            next_T[i] = (u64)sum;
            carry = (u64)(sum >> 64);
        }
        for (int i = 0; i < 8; i++) T[i] = next_T[i];
    }

    while (ge256(T, P)) {
        u64 R[4];
        sub256(R, T, P);
        for (int i = 0; i < 4; i++) T[i] = R[i];
    }

    for (int i = 0; i < 4; i++) out[i] = T[i];
}
