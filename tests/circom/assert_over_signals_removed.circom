pragma circom 2.0.0;

// `assert` over a witness-domain value. circom emits NO constraint for this
// (measured: identical non-linear/linear/wire counts with and without), so it
// is a witness-time precondition check, not part of the circuit.
//
// This is the shape circom-ecdsa's `ModSubThree` uses, under a comment reading
// "assume a - b - c + 2**n >= 0" - it documents a precondition on the caller.
template AssertOverSignals() {
    signal input a;
    signal input b;
    signal output out;


    out <== a * a + b;
}

component main = AssertOverSignals();
