pragma circom 2.0.0;
include "comparators.circom";
template T() {
    signal input a;
    signal input b;
    signal output out;
    component lt = LessThan(32);
    lt.in[0] <== a;
    lt.in[1] <== b;
    out <== lt.out;
}
component main = T();
