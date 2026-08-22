template Inner(n) {
    signal input x;
    signal output y;
    y <== x * n;
}
template T() {
    signal input a;
    signal output out;
    var u = a & 1;
    component c = Inner(u);
    c.x <== a;
    out <== c.y;
}
component main = T();
