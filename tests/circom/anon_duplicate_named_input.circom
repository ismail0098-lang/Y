pragma circom 2.1.0;
include "anon_lib.circom";
template M() { signal input i; signal output o; signal p, q; (p, q) <== AnonTwo()(a <== i, a <== i); o <== p; }
component main = M();
