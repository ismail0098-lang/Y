pragma circom 2.1.0;
include "anon_lib.circom";

// Named inputs, deliberately given in the OPPOSITE order to the declaration,
// so a lowering that ignored the names and zipped positionally would compute
// a different circuit rather than the same one.
template M() {
    signal input i;
    signal input j;
    signal output o;
    signal x, y;
    (x, y) <== AnonTwo()(b <== j, a <== i);
    o <== x + y;
}
component main = M();
