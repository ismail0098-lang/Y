pragma circom 2.0.0;
// The same sum, but reaching a `<==`. circom refuses this ("Non quadratic
// constraints are not allowed"), and so must Y - at the USE, naming the cause.
template QuadIntoConstraint() {
    signal input a; signal input b; signal input c; signal input d;
    signal output out;
    out <== a * b + c * d;
}
component main = QuadIntoConstraint();
