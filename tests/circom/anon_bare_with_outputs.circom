pragma circom 2.1.0;
include "anon_lib.circom";
// circom: "This expression must be a tuple or an anonymous component".
template M() { signal input i; signal output o; AnonTwo()(i, i); o <== i; }
component main = M();
