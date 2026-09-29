// An unknown index must not absorb a genuine typo. `nope` is not declared, and
// that is an error whether or not the index can be evaluated.
template T() {
    signal input a;
    signal output out;
    var u = a & 1;
    out <-- nope[u];
    out === out;
}
component main = T();
