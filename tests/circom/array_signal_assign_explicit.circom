pragma circom 2.1.0;
include "anon_lib.circom";
// Element by element. Must be the same artifact.
template M() {
    signal input i;
    signal input pair[3];
    signal output o;

    component fromLiteral = AnonSum(3);
    fromLiteral.in[0] <== i;
    fromLiteral.in[1] <== i + 1;
    fromLiteral.in[2] <== i + 2;

    component fromArray = AnonSum(3);
    fromArray.in[0] <== pair[0];
    fromArray.in[1] <== pair[1];
    fromArray.in[2] <== pair[2];

    o <== fromLiteral.out + fromArray.out;
}
component main = M();
