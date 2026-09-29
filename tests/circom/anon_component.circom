pragma circom 2.1.0;
include "anon_lib.circom";

template M() {
    signal input i;
    signal input j;
    signal output o;

    signal p, q;
    (p, q) <== AnonTwo()(i, j);
    signal s <== AnonSum(3)([i, j, p]);
    o <== q + s;
}

component main = M();
