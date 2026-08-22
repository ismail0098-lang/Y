template T() {
    signal input a;
    signal output out;
    signal mid;
    var u = a & 1;
    if (u == 1) {
        mid <== a * 2;
    }
    out <== a;
}
component main = T();
