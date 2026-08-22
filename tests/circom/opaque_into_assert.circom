template T() {
    signal input a;
    signal output out;
    var u = a & 1;
    assert(u);
    out <== a;
}
component main = T();
