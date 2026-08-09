pragma circom 2.0.0;
// A branch on a signal's VALUE decides which constraints exist, so it cannot
// be compiled. circom refuses this too.
template T() {
    signal input a;
    signal output out;
    if (a) { out <== 1; } else { out <== 2; }
}
component main = T();
