pragma circom 2.1.0;
include "anon_lib.circom";

// Byte-for-byte what `anon_component.circom` must desugar to. Hand-written, so
// a change in the desugaring shows up as an artifact difference rather than as
// a constraint count that happens to agree.
template M() {
    signal input i;
    signal input j;
    signal output o;

    signal p, q;
    component c0 = AnonTwo();
    c0.a <== i;
    c0.b <== j;
    p <== c0.x;
    q <== c0.y;

    signal s;
    component c1 = AnonSum(3);
    c1.in[0] <== i;
    c1.in[1] <== j;
    c1.in[2] <== p;
    s <== c1.out;

    o <== q + s;
}

component main = M();
