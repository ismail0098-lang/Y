// An unknown index on the LEFT of a `<==`. Which wire the constraint mentions
// would depend on a witness value, so this is refused AT THE INDEX -- unlike a
// read at an unknown index, which merely yields an unknown value.
// circom refuses it too, with error[T20462].
template T() {
    signal input a;
    signal output out[2];
    var u = a & 1;
    out[u] <== 1;
    out[1 - u] <== 2;
}
component main = T();
