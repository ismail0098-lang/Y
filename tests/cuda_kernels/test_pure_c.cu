
typedef unsigned long long u64;

__device__ void mod_mul_256(u64* res, const u64* a, const u64* b) {
    u64 K[4] = {0xbc1e0a6c0fffffffULL, 0xd7cc17b786468f6eULL, 0x47afba497e7ea7a2ULL, 0xcf9bb18d1ece5fd6ULL};
    u64 P[4] = {0x43e1f593f0000001ULL, 0x2833e84879b97091ULL, 0xb85045b68181585dULL, 0x30644e72e131a029ULL};

    u64 C[8] = {0};
    for (int i = 0; i < 4; i++) {
        u64 carry = 0;
        for (int j = 0; j < 4; j++) {
            unsigned __int128 prod = (unsigned __int128)a[i] * (unsigned __int128)b[j] + (unsigned __int128)C[i + j] + (unsigned __int128)carry;
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
                unsigned __int128 prod = (unsigned __int128)C[4 + i] * (unsigned __int128)K[j] + (unsigned __int128)CK[i + j] + (unsigned __int128)carry;
                CK[i + j] = (u64)prod;
                carry = (u64)(prod >> 64);
            }
            CK[i + 4] += carry;
        }

        u64 carry = 0;
        for (int i = 0; i < 4; i++) {
            unsigned __int128 sum = (unsigned __int128)C[i] + (unsigned __int128)CK[i] + (unsigned __int128)carry;
            C[i] = (u64)sum;
            carry = (u64)(sum >> 64);
        }
        for (int i = 4; i < 8; i++) {
            unsigned __int128 sum = (unsigned __int128)CK[i] + (unsigned __int128)carry;
            C[i] = (u64)sum;
            carry = (u64)(sum >> 64);
        }
    }

    for (int step = 0; step < 16; step++) {
        bool ge = false;
        if (C[4] > 0) {
            ge = true;
        } else {
            for (int i = 3; i >= 0; i--) {
                if (C[i] > P[i]) { ge = true; break; }
                if (C[i] < P[i]) { ge = false; break; }
            }
        }
        if (ge) {
            u64 borrow = 0;
            for (int i = 0; i < 4; i++) {
                unsigned __int128 diff = (unsigned __int128)C[i] - (unsigned __int128)P[i] - (unsigned __int128)borrow;
                C[i] = (u64)diff;
                borrow = (diff >> 127) & 1;
            }
            if (C[4] >= borrow) C[4] -= borrow; else C[4] = 0;
        }
    }

    res[0] = C[0]; res[1] = C[1]; res[2] = C[2]; res[3] = C[3];
}

extern "C" __global__ void test_pure_c_kernel(u64* out, const u64* A, const u64* B) {
    u64 a[4] = {A[0], A[1], A[2], A[3]};
    u64 b[4] = {B[0], B[1], B[2], B[3]};
    u64 r[4];

    mod_mul_256(r, a, b);

    out[0] = r[0]; out[1] = r[1]; out[2] = r[2]; out[3] = r[3];
}
