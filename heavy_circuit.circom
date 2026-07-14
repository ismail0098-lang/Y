pragma circom 2.0.0;

template HeavyCircuit() {
    signal input x;
    signal input y;
    signal output result;

    signal temp[1000001];
    temp[0] <== x;
    for (var i = 0; i < 1000000; i++) {
        temp[i+1] <== temp[i] * y;
    }
    result <== temp[1000000];
}

component main {public [x, y]} = HeavyCircuit();
