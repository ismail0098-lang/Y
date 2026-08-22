pragma circom 2.0.0;

// A constant chain deep enough that convergence, not just correctness, is
// tested.
//
// Each stage is quadratic until the PREVIOUS one has been pinned, which is the
// structure of circomlib's `WindowMulFix`: 2B..8B are each derived from the
// last by a `MontgomeryAdd` whose `<--` advice is opaque and whose `===` only
// becomes linear once its input is known.
//
// The pass therefore has to cascade through all eight stages. If it reads each
// constraint AS STORED rather than resolving it through the eliminations
// already made in the same round, one round advances the chain by one link and
// the whole thing hits the round cap with most of the circuit still symbolic.
template T() {
    signal input x;
    signal output out;
    var N = 8;
    signal a[N+1];
    signal lam[N+1];
    signal sq[N+1];

    a[0] <-- 4;
    3*a[0] === 12;

    for (var i = 1; i <= N; i++) {
        lam[i] <-- 2;
        lam[i] * a[i-1] === 2 * a[i-1];
        sq[i] <== lam[i] * lam[i];
        a[i] <== sq[i] * a[i-1];
    }

    out <== a[N] + x;
}

component main = T();
