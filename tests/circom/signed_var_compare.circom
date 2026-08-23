pragma circom 2.0.0;

// circom orders `var`s by their SIGNED representative: a value above (p-1)/2
// denotes `v - p`. Every output below is a case where that disagrees with the
// canonical (unsigned) order, plus two controls where the two agree.
//
// H  = (p-1)/2, the largest value circom reads as POSITIVE
// H1 = (p+1)/2, the most NEGATIVE value
template SignedVarCompare() {
    signal output o[9];

    var H  = 10944121435919637611123202872628637544274182200208017171849102093287904247808;
    var H1 = 10944121435919637611123202872628637544274182200208017171849102093287904247809;
    var M1 = 0 - 1;   // p-1, i.e. -1

    // Discriminating: signed and canonical give different answers.
    o[0] <== (M1 < 1)   ? 1 : 0;   // signed  -1 <  1  TRUE   | canonical FALSE
    o[1] <== (H  < H1)  ? 1 : 0;   // signed   H < -H  FALSE  | canonical TRUE
    o[2] <== (0  < M1)  ? 1 : 0;   // signed   0 < -1  FALSE  | canonical TRUE
    o[3] <== (M1 > 1)   ? 1 : 0;   // signed  -1 >  1  FALSE  | canonical TRUE
    o[4] <== (H1 < 0)   ? 1 : 0;   // signed  -H <  0  TRUE   | canonical FALSE
    o[5] <== (M1 >= 0)  ? 1 : 0;   // signed  -1 >= 0  FALSE  | canonical TRUE

    // Controls: both orders agree. These must not move.
    o[6] <== (M1 <= M1) ? 1 : 0;   // reflexive
    o[7] <== (H  > 0)   ? 1 : 0;   // positive vs zero
    o[8] <== (2  < 5)   ? 1 : 0;   // both small
}

component main = SignedVarCompare();
