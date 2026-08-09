pragma circom 2.0.0;

// `a` is defined by a linear constraint and then asserted to equal something
// else. No assignment of `x` satisfies both, so this circuit is unsatisfiable
// and must STAY unsatisfiable: the substitution pass deletes the constraint
// that defines `a`, and if it also let the `===` collapse away, an impossible
// statement would become a provable one.
template Contradiction() {
    signal input x;
    signal output out;

    signal a;
    a <== x + 1;
    a === x + 2;

    out <== a;
}

component main = Contradiction();
