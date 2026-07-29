
typedef unsigned long long u64;
typedef unsigned __int128 u128;

struct Limbs4 {
    u64 v[4];
};

__device__ Limbs4 mul4x4_limbs(Limbs4 A, Limbs4 B) {
    u64 C[8] = {0};
    for (int i = 0; i < 4; i++) {
        u64 carry = 0;
        for (int j = 0; j < 4; j++) {
            u128 prod = (u128)A.v[i] * (u128)B.v[j] + (u128)C[i + j] + (u128)carry;
            C[i + j] = (u64)prod;
            carry = (u64)(prod >> 64);
        }
        C[i + 4] += carry;
    }

    u64 K[4] = {0xbc1e0a6c0fffffffULL, 0xd7cc17b786468f6eULL, 0x47afba497e7ea7a2ULL, 0xcf9bb18d1ece5fd6ULL};
    u64 P[4] = {0x43e1f593f0000001ULL, 0x2833e84879b97091ULL, 0xb85045b68181585dULL, 0x30644e72e131a029ULL};

    u64 T[8];
    for (int i = 0; i < 8; i++) T[i] = C[i];

    while (T[4] > 0 || T[5] > 0 || T[6] > 0 || T[7] > 0) {
        Limbs4 T_hi = {T[4], T[5], T[6], T[7]};
        Limbs4 K_limbs = {K[0], K[1], K[2], K[3]};
        u64 CK[8] = {0};
        for (int i = 0; i < 4; i++) {
            u64 carry = 0;
            for (int j = 0; j < 4; j++) {
                u128 prod = (u128)T_hi.v[i] * (u128)K_limbs.v[j] + (u128)CK[i + j] + (u128)carry;
                CK[i + j] = (u64)prod;
                carry = (u64)(prod >> 64);
            }
            CK[i + 4] += carry;
        }

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

    while (true) {
        bool ge = true;
        for (int i = 3; i >= 0; i--) {
            if (T[i] > P[i]) { ge = true; break; }
            if (T[i] < P[i]) { ge = false; break; }
        }
        if (!ge) break;

        u64 borrow = 0;
        for (int i = 0; i < 4; i++) {
            u128 diff = (u128)T[i] - (u128)P[i] - (u128)borrow;
            T[i] = (u64)diff;
            borrow = (diff >> 127) & 1;
        }
    }

    Limbs4 res = {T[0], T[1], T[2], T[3]};
    return res;
}

extern "C" __global__ void test_entry_kernel(u64* out, const u64* x, const u64* y) {
    Limbs4 A = {x[0], x[1], x[2], x[3]};
    Limbs4 B = {y[0], y[1], y[2], y[3]};
    Limbs4 R = mul4x4_limbs(A, B);
    out[0] = R.v[0]; out[1] = R.v[1]; out[2] = R.v[2]; out[3] = R.v[3];
}
