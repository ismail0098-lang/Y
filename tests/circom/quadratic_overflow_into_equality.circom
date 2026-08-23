pragma circom 2.0.0;
// ...and via `===` rather than `<==`, which is a different code path.
template QuadIntoEquality() {
    signal input a; signal input b; signal input c; signal input d;
    signal output out;
    out <-- 0;
    a * b + c * d === out;
}
component main = QuadIntoEquality();
