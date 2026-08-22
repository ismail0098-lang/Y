pragma circom 2.0.0;

// The shape that `--O2` exploits on the whole `EscalarMulFix` family, reduced
// to its essence.
//
// `a` is assigned in the WITNESS domain (`<--`), so no constant folding at
// lowering time can see it. What pins it is the `===` beside it, whose right
// hand side is a constant: `3*a === 12`. That is `k*lin = c0`, the shape this
// pass used to skip because it only ever looked for a single wire in `C`.
//
// Once `a` is known the rest cascades: `b <== a*a` has two constant operands,
// `c <== b + a` is then constant too. circomlib does exactly this, at depth 8,
// by passing a fixed base point in as a signal and deriving 2B..8B from it.
template T() {
    signal input x;
    signal output out;
    signal a;
    signal b;
    signal c;

    a <-- 4;
    3*a === 12;

    b <== a*a;
    c <== b + a;

    out <== c + x;
}

component main = T();
