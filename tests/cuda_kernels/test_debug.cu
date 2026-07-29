
#include <stdio.h>
typedef unsigned long long u64;

extern "C" __global__ void test_debug(const u64* A, const u64* B, u64* out) {
    u64 a[4] = {A[0], A[1], A[2], A[3]};
    u64 b[4] = {B[0], B[1], B[2], B[3]};
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
        u64 idx = i + 4;
        while (carry > 0 && idx < 8) {
            unsigned __int128 sum = (unsigned __int128)C[idx] + (unsigned __int128)carry;
            C[idx] = (u64)sum;
            carry = (u64)(sum >> 64);
            idx++;
        }
    }

    for (int pass_num = 0; pass_num < 64; pass_num++) {
        u64 T_hi[4] = {C[4], C[5], C[6], C[7]};
        C[4] = 0; C[5] = 0; C[6] = 0; C[7] = 0;

        u64 CK[8] = {0};
        for (int i = 0; i < 4; i++) {
            u64 carry = 0;
            for (int j = 0; j < 4; j++) {
                unsigned __int128 prod = (unsigned __int128)T_hi[i] * (unsigned __int128)K[j] + (unsigned __int128)CK[i + j] + (unsigned __int128)carry;
                CK[i + j] = (u64)prod;
                carry = (u64)(prod >> 64);
            }
            u64 idx = i + 4;
            while (carry > 0 && idx < 8) {
                unsigned __int128 sum = (unsigned __int128)CK[idx] + (unsigned __int128)carry;
                CK[idx] = (u64)sum;
                carry = (u64)(sum >> 64);
                idx++;
            }
        }

        u64 carry = 0;
        for (int i = 0; i < 8; i++) {
            unsigned __int128 sum = (unsigned __int128)C[i] + (unsigned __int128)CK[i] + (unsigned __int128)carry;
            C[i] = (u64)sum;
            carry = (u64)(sum >> 64);
        }
    }

    for (int step = 0; step < 8; step++) {
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

    out[0] = C[0]; out[1] = C[1]; out[2] = C[2]; out[3] = C[3];
}

int main() {
    u64 x[4] = {0x43e1f593efffffffULL, 0x2833e84879b97091ULL, 0xb85045b68181585dULL, 0x30644e72e131a029ULL}; // p - 2
    u64 y[4] = {0x43e1f593effffffeULL, 0x2833e84879b97091ULL, 0xb85045b68181585dULL, 0x30644e72e131a029ULL}; // p - 3

    u64 *d_x, *d_y, *d_out;
    cudaMalloc(&d_x, 32);
    cudaMalloc(&d_y, 32);
    cudaMalloc(&d_out, 32);
    cudaMemcpy(d_x, x, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, 32, cudaMemcpyHostToDevice);

    test_debug<<<1,1>>>(d_x, d_y, d_out);
    cudaDeviceSynchronize();

    u64 res[4];
    cudaMemcpy(res, d_out, 32, cudaMemcpyDeviceToHost);
    printf("Result limbs: res[0]=%llu, res[1]=%llu, res[2]=%llu, res[3]=%llu\n", res[0], res[1], res[2], res[3]);
    return 0;
}
