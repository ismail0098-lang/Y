pragma circom 2.1.0;
include "anon_lib.circom";
template M() {
    signal input i;
    signal output o;
    component c = AnonSum(3);
    c.in <== [i, i + 1];
    o <== c.out;
}
component main = M();
