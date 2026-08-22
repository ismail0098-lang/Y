template Inner() {
    signal input x;
    signal output y;
    y <== x * x;
}
template T() {
    signal input a;
    signal output out;
    var u = a & 1;
    if (u == 1) {
        component c = Inner();
    }
    out <== a;
}
component main = T();
