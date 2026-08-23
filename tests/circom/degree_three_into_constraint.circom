pragma circom 2.0.0;
// Degree 3 in a constraint. circom refuses; so must Y.
template DegThree() {
    signal input a; signal input b; signal input c;
    signal output out;
    out <== a * b * c;
}
component main = DegThree();
