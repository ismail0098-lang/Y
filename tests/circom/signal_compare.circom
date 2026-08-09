pragma circom 2.0.0;
// `==` over signals has no R1CS form; it is a gadget (circomlib IsEqual).
template T() {
    signal input a;
    signal input b;
    signal output out;
    out <== a == b;
}
component main = T();
