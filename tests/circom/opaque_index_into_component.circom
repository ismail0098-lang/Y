// An unknown COMPONENT index. Which subcomponent gets driven decides which
// constraints exist, so this is refused at the index. circom: error[T20462].
template I() {
    signal input x;
    signal output y;
    y <== x * x;
}
template T() {
    signal input a;
    signal output out;
    var u = a & 1;
    component c[2];
    c[0] = I();
    c[1] = I();
    c[u].x <== 3;
    out <== c[0].y;
}
component main = T();
