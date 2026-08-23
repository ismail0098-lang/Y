pragma circom 2.1.0;
include "anon_lib.circom";
// circom: "The number of elements in both tuples does not coincide".
template M() { signal input i; signal output o; signal p, q, r; (p, q, r) <== AnonTwo()(i, i); o <== p; }
component main = M();
