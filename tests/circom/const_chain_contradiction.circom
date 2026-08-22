pragma circom 2.0.0;

// The same shape, over-determined. `a` cannot be both 4 and 3, so no witness
// exists. Constant propagation must not "simplify" this into satisfiability:
// it pins `a` from the first constraint and the second must then collapse to a
// non-zero constant identity, which `constraint_is_vacuous` keeps on purpose.
template T() {
    signal input x;
    signal output out;
    signal a;

    a <-- 4;
    3*a === 12;
    5*a === 15;

    out <== a + x;
}

component main = T();
