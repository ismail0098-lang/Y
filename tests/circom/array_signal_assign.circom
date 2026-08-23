pragma circom 2.1.0;
include "anon_lib.circom";
// A whole signal array driven at once, from an array LITERAL and from another
// signal array by name. semaphore writes `isLessThan.in <== [secret, l];`.
template M() {
    signal input i;
    signal input pair[3];
    signal output o;

    component fromLiteral = AnonSum(3);
    fromLiteral.in <== [i, i + 1, i + 2];

    component fromArray = AnonSum(3);
    fromArray.in <== pair;

    o <== fromLiteral.out + fromArray.out;
}
component main = M();
