pragma circom 2.0.0;
// `\=` is compound INTEGER division. circom's `\` is the quotient; `/` is
// field division by the modular inverse, and they agree only when the divisor
// divides exactly. zk-email's `log2Ceil` is written with `n \= 2;`.
function halves(a) {
    var n = a;
    var r = 0;
    while (n > 0) { r++; n \= 2; }
    return r;
}
template M() {
    signal input i;
    signal output o;
    o <== halves(100) * i;
}
component main = M();
