pragma circom 2.1.0;
include "anon_lib.circom";
// A dimensioned signal inside the declaring tuple, and `_` discarding one.
template Spread(n) {
    signal input a;
    signal output first;
    signal output rest[n];
    first <== a * a;
    for (var k = 0; k < n; k++) { rest[k] <== a + k; }
}
template M() {
    signal input i;
    signal output o;
    signal (f, r[3]) <== Spread(3)(i);
    o <== f + r[0] + r[2];
}
component main = M();
