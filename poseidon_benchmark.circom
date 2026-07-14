pragma circom 2.0.0;
include "circomlib/circuits/poseidon.circom";

template PoseidonBenchmark() {
    signal input x;
    signal input y;
    signal output out;

    component pos = Poseidon(2);
    pos.inputs[0] <== x;
    pos.inputs[1] <== y;
    out <== pos.out;
}

component main {public [x, y]} = PoseidonBenchmark();
