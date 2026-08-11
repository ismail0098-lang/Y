pragma circom 2.0.0;
include "comparators.circom";

// `IsEqual` is `IsZero(in[1] - in[0])`, so this is the same gadget one level up
// -- and it exercises the `==` direction of the ternary, where `IsZero` alone
// only exercises `!=`.
component main = IsEqual();
