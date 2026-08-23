pragma circom 2.1.0;
include "anon_lib.circom";
// circom: "The number of template input signals must coincide with the number
// of input parameters".
template M() { signal input i; signal output o; signal p, q; (p, q) <== AnonTwo()(i); o <== p; }
component main = M();
