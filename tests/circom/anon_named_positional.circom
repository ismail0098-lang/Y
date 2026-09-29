pragma circom 2.1.0;
include "anon_lib.circom";
// The positional spelling of `anon_named_inputs.circom`: a <== i, b <== j.
// The two must serialize identically.
template M() {
    signal input i;
    signal input j;
    signal output o;
    signal x, y;
    (x, y) <== AnonTwo()(i, j);
    o <== x + y;
}
component main = M();
