pragma circom 2.1.0;
include "anon_lib.circom";
// An anonymous component with no outputs, used as a whole statement. This is
// how circom-rln writes `RangeCheck(LIMIT_BIT_SIZE)(messageId, limit);`.
template M() {
    signal input i;
    signal output o;
    AnonBool()(i);
    o <== i + 1;
}
component main = M();
