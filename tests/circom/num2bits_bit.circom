pragma circom 2.0.0;
include "bitify.circom";
// Returns bit 3 of the input, so a wrong bit ORDER is visible in the value
// rather than only as unsatisfiability.
template T() {
    signal input x;
    signal output out;
    component b = Num2Bits(16);
    b.in <== x;
    out <== b.out[3];
}
component main = T();
