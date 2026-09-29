template T() {
    signal input a;
    signal output out;
    signal keep;
    var u = a & 1;
    var v = 0;
    while (u != 0) {
        v = v + 1;
        u = u - 1;
    }
    keep <== a * a;
    out <-- v;
}
component main = T();
