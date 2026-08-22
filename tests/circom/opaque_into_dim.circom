template T() {
    signal input a;
    signal output out;
    var u = a & 1;
    var t[u];
    out <== a;
}
component main = T();
