// The `<--` is opaque, but `out === a` determines `out`, so the witness solves.
// This is Sha256's shape in miniature.
template T() {
    signal input a;
    signal output out;
    var u = a & 1;
    out <-- u;
    out === a;
}
component main = T();
