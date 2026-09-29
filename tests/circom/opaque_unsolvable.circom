// The `<--` is opaque and the constraint is a SQUARE ROOT, which no
// back-propagation recovers. Satisfiability must come back false rather than
// the solver's default zero being passed off as a witness.
template T() {
    signal input a;
    signal output out;
    var u = a & 1;
    out <-- u;
    out * out === a;
}
component main = T();
