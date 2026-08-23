pragma circom 2.0.0;

// The other face of the signed-comparison bug, and the one that blocks real
// circuits: `j--` past zero reaches p-1, which is `>= 0` under the canonical
// order, so the loop never terminates. circom reads p-1 as -1 and stops.
//
// Before the fix this hit the 20,000,000-iteration guard. It is the shape
// `circom-ecdsa` and `zk-email/rsa` are built on.
template DescendingLoop() {
    signal output out;

    var acc = 0;
    for (var j = 3; j >= 0; j--) {
        acc = acc * 10 + j;
    }

    out <== acc;   // 3210
}

component main = DescendingLoop();
