// An unknown index in the WITNESS domain: reading a `var` array and a signal
// array at an index the compiler cannot evaluate, and storing into a `var`
// array at one. circom accepts all three, and none of them emits a constraint.
//
// This is zk-email's `long_div` in miniature -- a loop whose condition depends
// on a signal makes the loop counter unknown, and the counter is then an index.
template T() {
    signal input a;
    signal input in[4];
    signal output sq;
    signal output prod;

    var t[4] = [10, 20, 30, 40];
    var u = a & 1;

    t[u] = 99;

    signal fromVar;
    signal fromSignal;
    fromVar <-- t[0];
    fromSignal <-- in[u];

    // The author's own constraints. These are the only ones in the circuit,
    // and they are what the count is compared against.
    sq <== fromVar * fromVar;
    prod <== fromVar * fromSignal;
}
component main = T();
