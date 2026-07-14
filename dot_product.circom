pragma circom 2.0.0;

template DotProduct() {
    signal input x;
    signal input y;
    signal output result;

    signal a[100001];
    signal b[100001];
    signal prod[100000];
    signal sum[100001];

    a[0] <== x;
    b[0] <== y;
    sum[0] <== 0;

    for (var i = 0; i < 100000; i++) {
        a[i+1] <== a[i] + 1;
        b[i+1] <== b[i] + 1;
        prod[i] <== a[i+1] * b[i+1];
        sum[i+1] <== sum[i] + prod[i];
    }

    result <== sum[100000];
}

component main {public [x, y]} = DotProduct();
