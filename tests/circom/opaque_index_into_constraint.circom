pragma circom 2.0.0;
// The same opaque array element, but reaching a `<==`. Must be refused.
function pick(s) {
    var r[2];
    if (s > 0) { r[0] = 1; r[1] = 2; return r; }
    r[0] = 3; r[1] = 4; return r;
}
template OpaqueIndexIntoConstraint() {
    signal input a;
    signal output out;
    var v[2] = pick(a);
    out <== v[0];
}
component main = OpaqueIndexIntoConstraint();
