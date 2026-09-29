pragma circom 2.1.0;
include "anon_lib.circom";
// circom: "This is the anonymous component whose use is not allowed".
template M() { signal input i; signal output o; o <== AnonSum(1)([i]) + 3; }
component main = M();
