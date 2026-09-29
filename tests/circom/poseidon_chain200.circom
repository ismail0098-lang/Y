pragma circom 2.0.0;
// A 200-hash Poseidon chain: 55,608 constraints, 55,611 wires after compaction.
//
// The size `zk_wire_compaction.rs` measures Groth16 at - big enough that setup
// and prove are hundreds of milliseconds rather than tens, so the ratio between
// two runs means something. circom compiles the same source to 103,400
// constraints and 103,403 wires.
include "poseidon.circom";

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

component main = Chain(200);
