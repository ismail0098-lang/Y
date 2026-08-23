pragma circom 2.1.0;
include "anon_lib.circom";
// The declaration and the drive written separately. Must be the same artifact;
// circom's own two forms are byte-identical too.
template M() {
    signal input i;
    signal input j;
    signal output o;
    signal p;
    signal q;
    (p, q) <== AnonTwo()(i, j);
    o <== p + q;
}
component main = M();
