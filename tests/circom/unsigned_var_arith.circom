pragma circom 2.0.0;

// circom's `\`, `%` and `>>` operate on the UNSIGNED canonical value, even
// where `<` on the same operand would read it as negative. Probed against
// circom 2.2.3 at `p-1`. This is the control that stops "make the whole front
// end signed" from passing.
template UnsignedVarArith() {
    signal output o[3];

    var M1 = 0 - 1;   // p-1, which `<` reads as -1

    o[0] <== M1 \ 2;    // unsigned (p-1)/2   | signed would be 0 or -1
    o[1] <== M1 % 2;    // 0                  | signed would be +/-1
    o[2] <== M1 >> 1;   // unsigned (p-1)/2   | signed would be -1
}

component main = UnsignedVarArith();
