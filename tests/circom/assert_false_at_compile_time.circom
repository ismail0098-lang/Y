pragma circom 2.0.0;

// A COMPILE-TIME false assert must still be a hard error. This is the control
// that stops "ignore every assert" from passing the file: the witness-domain
// relaxation must not reach a value the compiler can actually evaluate.
template AssertFalse() {
    signal input a;
    signal output out;

    assert(2 + 2 == 5);

    out <== a;
}

component main = AssertFalse();
