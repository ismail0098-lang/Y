pragma circom 2.0.0;

// `secp256k1_addunequal_func`'s shape, minimised.
//
// The branch depends on a SIGNAL, so it cannot be evaluated: the body is
// havoc'd and the function returns an opaque value - a SCALAR - where the
// caller declared an array. Indexing that is the opaque value propagating, not
// the user error "too many indices".
//
// `out` is `<--` from the advice and BOUND by the `===` below, which is what
// makes the witness solvable - the same structure as circomlib's Sha256.
function pick(s) {
    var r[2];
    if (s > 0) {
        r[0] = 1;
        r[1] = 2;
        return r;
    }
    r[0] = 3;
    r[1] = 4;
    return r;
}

template OpaqueArrayIndexAdvice() {
    signal input a;
    signal output out;

    var v[2] = pick(a);

    out <-- v[0];
    out === a * a;
}

component main = OpaqueArrayIndexAdvice();
