pragma circom 2.1.0;
include "anon_lib.circom";
// circom: "Anonymous components only admit the use of the operator <==".
template M() { signal input i; signal output o; o <-- AnonSum(1)([i]); o === o; }
component main = M();
