pragma circom 2.1.0;
include "anon_lib.circom";
// `_` discards an output. The component is still instantiated and its inputs
// still constrained -- only the binding is dropped.
template M() {
    signal input i;
    signal output o;
    signal keep;
    (keep, _) <== AnonTwo()(i, i);
    o <== keep;
}
component main = M();
