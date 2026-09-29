pragma circom 2.1.0;
include "anon_lib.circom";
// circom: "Left-side of the statement is not a tuple".
template M() { signal input i; signal output o; o <== AnonTwo()(i, i); }
component main = M();
