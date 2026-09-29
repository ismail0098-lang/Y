template T() {
    signal input a;
    signal output out;
    var t[4] = [1, 2, 3, 4];
    var u = a & 1;
    out <== t[u];
}
component main = T();
