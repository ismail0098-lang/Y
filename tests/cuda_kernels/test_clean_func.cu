
typedef unsigned long long u64;
typedef unsigned __int128 u128;

__device__ bool ge256(const u64* A, const u64* B) {
    if (A[7] > 0 || A[6] > 0 || A[5] > 0 || A[4] > 0) return true;
    for (int i = 3; i >= 0; i--) {
        if (A[i] > B[i]) return true;
        if (A[i] < B[i]) return false;
    }
    return true;
}

__device__ void sub256(u64* A, const u64* B) {
    u64 borrow = 0;
    for (int i = 0; i < 4; i++) {
        u128 diff = (u128)A[i] - (u128)B[i] - (u128)borrow;
        A[i] = (u64)diff;
        borrow = (diff >> 127) & 1;
    }
    if (A[4] >= borrow) {
        A[4] -= borrow;
    } else {
        A[4] = 0;
    }
}

extern "C" __device__ void mod_mul_256(u64* res, const u64* a, const u64* b) {
    u64 K[4] = {0xbc1e0a6c0fffffffULL, 0xd7cc17b786468f6eULL, 0x47afba497e7ea7a2ULL, 0xcf9bb18d1ece5fd6ULL};
    u64 P[4] = {0x43e1f593f0000001ULL, 0x2833e84879b97091ULL, 0xb85045b68181585dULL, 0x30644e72e131a029ULL};

    u64 C[8] = {0};
    for (int i = 0; i < 4; i++) {
        u64 carry = 0;
        for (int j = 0; j < 4; j++) {
            u128 prod = (u128)a[i] * (u128)b[j] + (u128)C[i + j] + (u128)carry;
            C[i + j] = (u64)prod;
            carry = (u64)(prod >> 64);
        }
        C[i + 4] += carry;
    }

    for (int pass_num = 0; pass_num < 64; pass_num++) {
        u64 CK[8] = {0};
        for (int i = 0; i < 4; i++) {
            u64 carry = 0;
            for (int j = 0; j < 4; j++) {
                u128 prod = (u128)C[4 + i] * (u128)K[j] + (u128)CK[i + j] + (u128)carry;
                CK[i + j] = (u64)prod;
                carry = (u64)(prod >> 64);
            }
            CK[i + 4] += carry;
        }

        u64 carry = 0;
        for (int i = 0; i < 4; i++) {
            u128 sum = (u128)C[i] + (u128)CK[i] + (u128)carry;
            C[i] = (u64)sum;
            carry = (u64)(sum >> 64);
        }
        for (int i = 4; i < 8; i++) {
            u128 sum = (u128)CK[i] + (u128)carry;
            C[i] = (u64)sum;
            carry = (u64)(sum >> 64);
        }
    }

    while (ge256(C, P)) {
        sub256(C, P);
    }

    res[0] = C[0]; res[1] = C[1]; res[2] = C[2]; res[3] = C[3];
}

extern "C" __global__ void test_witness_kernel(u64* witness_buf) {
    u64 x[4] = {witness_buf[4], witness_buf[5], witness_buf[6], witness_buf[7]}; // signal 1 (input x)
    u64 y[4] = {witness_buf[8], witness_buf[9], witness_buf[10], witness_buf[11]}; // signal 2 (input y)
    u64 r[4];

    mod_mul_256(r, x, y);

    witness_buf[12] = r[0]; witness_buf[13] = r[1]; witness_buf[14] = r[2]; witness_buf[15] = r[3]; // signal 3 (out)
}
