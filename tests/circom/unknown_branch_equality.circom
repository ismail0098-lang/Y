template T() {
    signal input a;
    signal output out;
    var u = a & 1;
    if (u == 1) {
        a * a === a;
    }
    out <== a;
}
component main = T();
