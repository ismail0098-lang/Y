function pick(n) {
    if (n == 0) {
        return 0;
    }
    return n * 2;
}
template T() {
    signal input a;
    signal output out;
    var u = a & 1;
    out <== pick(u);
}
component main = T();
