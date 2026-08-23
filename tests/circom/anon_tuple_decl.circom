pragma circom 2.1.0;
include "anon_lib.circom";
// `signal (a, b) <== T()(x)` declares the signals and drives them in one
// statement. zk-email's `email-verifier.circom` is written this way:
//   signal (bhRegexMatch, bhReveal[maxHeadersLength]) <== BodyHashRegex(n)(h);
template M() {
    signal input i;
    signal input j;
    signal output o;
    signal (p, q) <== AnonTwo()(i, j);
    o <== p + q;
}
component main = M();
