pragma circom 2.0.0;

// An ARRAY LITERAL as a template argument. circomlib passes curve base points
// this way -- `EscalarMul(8, [x, y])` and `EscalarMulFix(256, BASE8)` -- and the
// top-level `component main` path used to evaluate its arguments as scalars,
// so the construct worked for every nested component and failed only where a
// user writes it.
//
// The weights are distinct and asymmetric on purpose: an array that arrived
// empty, truncated, or reversed still yields a satisfiable circuit, so the test
// that goes with this checks the VALUES, not that it compiled.
template Weighted(n, w) {
    signal input in[n];
    signal output out;
    var acc = 0;
    for (var i = 0; i < n; i++) {
        acc += w[i] * in[i];
    }
    out <== acc;
}

component main = Weighted(3, [2, 30, 500]);
