// circomlib's `sqrt` in miniature: a function that branches on its argument
// and returns from inside the branch. The argument is a witness value, so no
// branch can be taken and the call has no compile-time result.
function pick(n) {
    if (n == 0) {
        return 0;
    }
    return n * 2;
}
template T() {
    signal input a;
    signal output out;
    signal keep;
    var u = a & 1;
    keep <== a * a;
    out <-- pick(u);
}
component main = T();
