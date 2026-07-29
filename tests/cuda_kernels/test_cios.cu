
typedef unsigned long long u64;
typedef unsigned __int128 u128;

__device__ void cios_mont_mul(u64* res, const u64* a, const u64* b) {
    u64 P[4] = {0x43e1f593f0000001ULL, 0x2833e84879b97091ULL, 0xb85045b68181585dULL, 0x30644e72e131a029ULL};
    u64 p_prime = 0xc2e1f593efffffffULL;

    u64 t[5] = {0};

    for (int i = 0; i < 4; i++) {
        // Step 1: Accumulate a[i] * b[0..3] into t[0..4]
        u64 carry = 0;
        for (int j = 0; j < 4; j++) {
            u128 prod = (u128)a[i] * (u128)b[j] + (u128)t[j] + (u128)carry;
            t[j] = (u64)prod;
            carry = (u64)(prod >> 64);
        }
        u128 sum_t = (u128)t[4] + (u128)carry;
        t[4] = (u64)sum_t;
        u64 t5 = (u64)(sum_t >> 64);

        // Step 2: Compute m = (t[0] * p_prime) mod 2^64
        u64 m = (u64)(t[0] * p_prime);

        // Step 3: Add m * P[0..3] to t[0..4]
        carry = 0;
        u128 prod_m = (u128)m * (u128)P[0] + (u128)t[0] + (u128)carry;
        carry = (u64)(prod_m >> 64); // t[0] addition term becomes 0 mod 2^64!

        for (int j = 1; j < 4; j++) {
            u128 prod = (u128)m * (u128)P[j] + (u128)t[j] + (u128)carry;
            t[j - 1] = (u64)prod;
            carry = (u64)(prod >> 64);
        }
        u128 sum_final = (u128)t[4] + (u128)carry;
        t[3] = (u64)sum_final;
        t[4] = t5 + (u64)(sum_final >> 64);
    }

    // Step 4: Conditional subtract P
    bool ge = false;
    if (t[4] > 0) {
        ge = true;
    } else {
        for (int i = 3; i >= 0; i--) {
            if (t[i] > P[i]) { ge = true; break; }
            if (t[i] < P[i]) { ge = false; break; }
        }
    }

    if (ge) {
        u64 borrow = 0;
        for (int i = 0; i < 4; i++) {
            u128 diff = (u128)t[i] - (u128)P[i] - (u128)borrow;
            t[i] = (u64)diff;
            borrow = (diff >> 127) & 1;
        }
    }

    res[0] = t[0]; res[1] = t[1]; res[2] = t[2]; res[3] = t[3];
}

extern "C" __global__ void test_cios_kernel(u64* out, const u64* A, const u64* B) {
    u64 a[4] = {A[0], A[1], A[2], A[3]};
    u64 b[4] = {B[0], B[1], B[2], B[3]};
    u64 r[4];

    cios_mont_mul(r, a, b);

    out[0] = r[0]; out[1] = r[1]; out[2] = r[2]; out[3] = r[3];
}
