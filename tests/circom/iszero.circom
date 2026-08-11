pragma circom 2.0.0;
include "comparators.circom";

// circomlib's IsZero, unmodified. Its witness advice is
// `inv <-- in != 0 ? 1/in : 0` -- a ternary over a signal, with a division by
// the very value the branch tests against zero. Y refused this until
// `WitnessOp::IfZeroLc` existed, and refusing it cost seven circomlib circuits:
// IsEqual, ForceEqualIfEnabled, both SMT circuits, both EdDSA verifiers and
// Multiplexer are all built on it.
component main = IsZero();
