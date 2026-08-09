pragma circom 2.0.0;

// Deliberately written the way real circom is: every step of the affine part
// gets its own named signal and its own `<==`. Each of those is a constraint
// that carries no information a verifier needs - which is exactly what the
// linear-substitution pass exists to remove.
//
// The two `* ` steps are genuinely quadratic and must survive.
template LinearChain(n) {
    signal input x;
    signal output out;

    signal step[n];
    step[0] <== x + 1;
    for (var i = 1; i < n; i++) {
        step[i] <== step[i-1] + 2 * x + i;
    }

    signal sq;
    sq <== step[n-1] * step[n-1];

    signal shifted;
    shifted <== sq + 7;

    out <== shifted * x;
}

component main = LinearChain(8);
