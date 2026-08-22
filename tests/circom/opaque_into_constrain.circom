template T() {
    signal input a;
    signal output out;
    var u = a & 1;
    out <== u;
}
component main = T();
