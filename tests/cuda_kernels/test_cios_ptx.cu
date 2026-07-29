
typedef unsigned long long u64;
typedef unsigned __int128 u128;

extern "C" __global__ void test_cios_ptx(u64* witness_buf) {
    u64 a[4] = {witness_buf[4], witness_buf[5], witness_buf[6], witness_buf[7]}; // signal 1
    u64 b[4] = {witness_buf[8], witness_buf[9], witness_buf[10], witness_buf[11]}; // signal 2

    u64 P[4] = {0x43e1f593f0000001ULL, 0x2833e84879b97091ULL, 0xb85045b68181585dULL, 0x30644e72e131a029ULL};
    u64 R2[4] = {0x1bb8e645ae216da7ULL, 0x53fe3ab1e35c59e3ULL, 0x8c49833d53bb8085ULL, 0x0216d0b17f4e44a5ULL};
    u64 p_prime = 0xc2e1f593efffffffULL;

    u64 tmp[4];
    u64 r[4];

    // Pass 1: tmp = cios(a, b)
    {
        u64 t[5] = {0};
        for (int i = 0; i < 4; i++) {
            u64 carry = 0;
            for (int j = 0; j < 4; j++) {
                u128 prod = (u128)a[i] * (u128)b[j] + (u128)t[j] + (u128)carry;
                t[j] = (u64)prod;
                carry = (u64)(prod >> 64);
            }
            u128 sum_t = (u128)t[4] + (u128)carry;
            t[4] = (u64)sum_t;
            u64 t5 = (u64)(sum_t >> 64);

            u64 m = (u64)(t[0] * p_prime);

            carry = 0;
            u128 prod_m = (u128)m * (u128)P[0] + (u128)t[0] + (u128)carry;
            carry = (u64)(prod_m >> 64);

            for (int j = 1; j < 4; j++) {
                u128 prod = (u128)m * (u128)P[j] + (u128)t[j] + (u128)carry;
                t[j - 1] = (u64)prod;
                carry = (u64)(prod >> 64);
            }
            u128 sum_final = (u128)t[4] + (u128)carry;
            t[3] = (u64)sum_final;
            t[4] = t5 + (u64)(sum_final >> 64);
        }

        bool ge = false;
        if (t[4] > 0) ge = true;
        else {
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
        tmp[0] = t[0]; tmp[1] = t[1]; tmp[2] = t[2]; tmp[3] = t[3];
    }

    // Pass 2: r = cios(tmp, R2)
    {
        u64 t[5] = {0};
        for (int i = 0; i < 4; i++) {
            u64 carry = 0;
            for (int j = 0; j < 4; j++) {
                u128 prod = (u128)tmp[i] * (u128)R2[j] + (u128)t[j] + (u128)carry;
                t[j] = (u64)prod;
                carry = (u64)(prod >> 64);
            }
            u128 sum_t = (u128)t[4] + (u128)carry;
            t[4] = (u64)sum_t;
            u64 t5 = (u64)(sum_t >> 64);

            u64 m = (u64)(t[0] * p_prime);

            carry = 0;
            u128 prod_m = (u128)m * (u128)P[0] + (u128)t[0] + (u128)carry;
            carry = (u64)(prod_m >> 64);

            for (int j = 1; j < 4; j++) {
                u128 prod = (u128)m * (u128)P[j] + (u128)t[j] + (u128)carry;
                t[j - 1] = (u64)prod;
                carry = (u64)(prod >> 64);
            }
            u128 sum_final = (u128)t[4] + (u128)carry;
            t[3] = (u64)sum_final;
            t[4] = t5 + (u64)(sum_final >> 64);
        }

        bool ge = false;
        if (t[4] > 0) ge = true;
        else {
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
        r[0] = t[0]; r[1] = t[1]; r[2] = t[2]; r[3] = t[3];
    }

    witness_buf[12] = r[0]; witness_buf[13] = r[1]; witness_buf[14] = r[2]; witness_buf[15] = r[3];
}
