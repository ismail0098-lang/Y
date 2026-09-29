// Shared by the anonymous-component fixtures. Two inputs, two outputs, and an
// array input, so the desugaring is exercised on more than a scalar.
template AnonTwo() {
    signal input a;
    signal input b;
    signal output x;
    signal output y;
    x <== a * b;
    // Deliberately ASYMMETRIC in a and b, so that a lowering which ignored
    // the input NAMES and zipped positionally would compute a different
    // circuit rather than the same one.
    y <== a + 2 * b;
}

template AnonSum(n) {
    signal input in[n];
    signal output out;
    var s = 0;
    for (var k = 0; k < n; k++) { s += in[k]; }
    out <== s;
}

// No outputs: legal as a bare statement, which is how circom-rln's RangeCheck
// is written.
template AnonBool() {
    signal input a;
    a * (a - 1) === 0;
}
