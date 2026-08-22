pragma circom 2.0.0;
include "bitify.circom";

// The regression for the cycle this optimisation can create.
//
// Num2Bits' recomposition is `1 * (sum 2^i b_i - in) = 0` - a linear equation
// with an EMPTY `C`. Read as a definition of `in`, it says `in` is the sum of
// the bits; but every bit's witness recipe decomposes `in`, so the recipes end
// up deriving their values from a value derived from themselves. The circuit
// stays satisfiable and no witness can be found.
template T() {
    signal input x;
    signal output out;
    component n = Num2Bits(16);
    n.in <== x;
    signal acc[17];
    acc[0] <== 0;
    for (var i = 0; i < 16; i++) {
        acc[i+1] <== acc[i] + n.out[i] * (1 << i);
    }
    out <== acc[16];
}

component main = T();
