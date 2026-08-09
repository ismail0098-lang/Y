pragma circom 2.0.0;
template T() {
    signal input a;
    signal input b;
    signal input c;
    signal output out;
    out <== a * b * c;
}
component main = T();
