pragma circom 2.0.0;
// A genuine mis-index of a KNOWN scalar. Nothing about the opaque relaxation
// should make this compile.
template TooMany() {
    signal output out;
    var x = 5;
    out <== x[0];
}
component main = TooMany();
