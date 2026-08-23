pragma circom 2.0.0;
// Array inputs, a public input, a private one, and two outputs -- the four
// things a circom `input.json` has to be able to express. `sum` is the whole
// array so a dropped element shows up; `scaled` reads the public input, so a
// public input left at zero (which is what Y did before `pub_in` was bound)
// makes it wrong rather than merely missing.
template M() {
    signal input scale;
    signal input in[3];
    signal input nested[2][2];
    signal output sum;
    signal output scaled;

    var s = 0;
    for (var i = 0; i < 3; i++) { s += (i + 1) * in[i]; }
    for (var i = 0; i < 2; i++) {
        for (var j = 0; j < 2; j++) { s += (10 * (i + 1) + j) * nested[i][j]; }
    }
    sum <== s;
    scaled <== scale * in[0];
}
component main {public [scale]} = M();
