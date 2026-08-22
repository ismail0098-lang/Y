template T() {
    signal input a;
    signal output out;
    var u = a & 1;
    var v = 7;
    if (u == 1) {
        v = 9;
    }
    out <== v;
}
component main = T();
