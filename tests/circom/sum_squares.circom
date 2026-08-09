pragma circom 2.0.0;

// Exercises subcomponents, component arrays, signal arrays, compile-time
// `for`/`if`, and a template parameter.
template Square() {
    signal input x;
    signal output y;
    y <== x * x;
}

template SumSquares(n) {
    signal input in[n];
    signal output out;
    component sq[n];
    signal partial[n];
    for (var i = 0; i < n; i++) {
        sq[i] = Square();
        sq[i].x <== in[i];
        if (i == 0) {
            partial[i] <== sq[i].y;
        } else {
            partial[i] <== partial[i-1] + sq[i].y;
        }
    }
    out <== partial[n-1];
}

component main = SumSquares(4);
