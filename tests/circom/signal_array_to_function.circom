// A signal ARRAY passed to a circom `function`, weighted by position so the
// order is observable, and returned into a real constraint.
function weighted(v, n) {
    var s = 0;
    for (var i = 0; i < n; i++) {
        s = s + v[i] * (i + 1);
    }
    return s;
}
template T() {
    signal input x[4];
    signal output out;
    out <== weighted(x, 4);
}
component main = T();
