pragma circom 2.0.0;

// `BigMult`'s shape, minimised: a `var` accumulates a sum of products of
// SIGNALS - more than one R1CS constraint can hold - feeds it to a `<--` as
// advice, and the author's own constraints bind the result.
//
// The sum is witness-domain: it never appears in a constraint, so refusing the
// arithmetic refused a legal circuit. The `===` below is what makes `out`
// determined, which is why the witness still solves.
template QuadraticOverflowAdvice() {
    signal input a;
    signal input b;
    signal input c;
    signal input d;
    signal output out;

    var acc = a * b + c * d;   // exceeds Quad -> witness-domain
    out <-- acc;

    signal t1;
    signal t2;
    t1 <== a * b;
    t2 <== c * d;
    out === t1 + t2;           // ...and this is what binds it
}

component main = QuadraticOverflowAdvice();
