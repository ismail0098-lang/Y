pragma circom 2.1.0;
include "anon_lib.circom";
// a <== j, b <== i -- the operands the other way round. `AnonTwo` is
// asymmetric in b, so this is a DIFFERENT circuit, and it is what a lowering
// that ignored the input names and zipped positionally would produce for
// `anon_named_inputs.circom`.
template M() {
    signal input i;
    signal input j;
    signal output o;
    signal x, y;
    (x, y) <== AnonTwo()(j, i);
    o <== x + y;
}
component main = M();
