// A store at an unknown index makes the WHOLE array unknown: some element
// changed and there is no way to say which. So `t[3]` below is unknown even
// though 3 is a constant, and it may not reach a `<==`.
//
// Without that, `t[3]` would still read 40 -- a constraint built from a value
// the program may have overwritten. circom refuses this too.
template T() {
    signal input a;
    signal output out;
    var t[4] = [10, 20, 30, 40];
    var u = a & 1;
    t[u] = 99;
    out <== t[3];
}
component main = T();
