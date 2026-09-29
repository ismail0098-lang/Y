pragma circom 2.0.0;
include "bitify.circom";
// Decompose, then recompose with the SAME weights. If any bit is wrong the
// Num2Bits recomposition constraint (`lc1 === in`) is unsatisfiable, so this
// is a real check on the `<--` witness recipe and not just on the arithmetic.
template T() {
    signal input x;
    signal output out;
    component b = Num2Bits(16);
    b.in <== x;
    signal acc[17];
    acc[0] <== 0;
    for (var i = 0; i < 16; i++) {
        acc[i+1] <== acc[i] + b.out[i] * (1 << i);
    }
    out <== acc[16];
}
component main = T();
