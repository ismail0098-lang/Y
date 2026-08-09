pragma circom 2.0.0;
include "poseidon.circom";

// Big enough that Groth16 setup and prove dominate their own fixed costs, small
// enough to run in a test. Used by `zk_linear_substitution.rs` to measure what
// the constraint reduction is actually worth downstream.
template Chain(n) {
    signal input x;
    signal input y;
    signal output out;
    component h[n];
    signal acc[n+1];
    acc[0] <== x;
    for (var i = 0; i < n; i++) {
        h[i] = Poseidon(2);
        h[i].inputs[0] <== acc[i];
        h[i].inputs[1] <== y;
        acc[i+1] <== h[i].out;
    }
    out <== acc[n];
}

component main = Chain(20);
