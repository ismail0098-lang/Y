pragma circom 2.0.0;

// `b` is declared and never used. circom keeps it: the number and order of a
// circuit's inputs is part of the statement it proves, so dropping an unused one
// changes the interface a verifier was compiled against. Used by
// `zk_wire_compaction.rs` to pin that compaction does not "collect" it.
template Unused() {
    signal input a;
    signal input b;
    signal output out;
    out <== a * a;
}

component main = Unused();
