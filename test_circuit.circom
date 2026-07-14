pragma circom 2.0.0;

include "circomlib/circuits/comparators.circom";

template TestCircuit() {
    signal input x;
    signal input y;
    signal output result;

    // 1. Non-linear multiplication constraint: A * B = C
    signal product;
    product <== x * y;

    // 2. Loop unrolling: loop_sum = loop_sum + x * i
    // loop_sum = 0 + 0*x + 1*x + 2*x + 3*x + 4*x = 10 * x
    signal loop_sum;
    loop_sum <== 10 * x;

    // 3. Conditional multiplexer (IsEqual):
    // if x == y { cond_val = 100 } else { cond_val = 200 }
    // Using Circom's standard library comparator template:
    component eq = IsEqual();
    eq.in[0] <== x;
    eq.in[1] <== y;

    // Multiplexer selector logic: cond_val = eq * 100 + (1 - eq) * 200
    signal cond_val;
    cond_val <== eq.out * 100 + (1 - eq.out) * 200;

    // Return the final constraint outcome
    result <== product + loop_sum + cond_val;
}

component main {public [x, y]} = TestCircuit();
