// Both arms return a CONSTANT, so a lowering that ignores the `return` inside
// the unevaluable branch falls through and produces a confident 7 - a wrong
// circuit rather than a refusal. The unknown must survive to the `<==`.
function pick(n) {
    if (n == 0) {
        return 5;
    }
    return 7;
}
template T() {
    signal input a;
    signal output out;
    var u = a & 1;
    out <== pick(u);
}
component main = T();
