template T() {
    signal input a;
    signal output out;
    signal keep;
    var u = a & 1;
    var v = 7;
    if (u == 1) {
        v = 9;
    }
    keep <== a * a;
    out <-- v;
}
component main = T();
