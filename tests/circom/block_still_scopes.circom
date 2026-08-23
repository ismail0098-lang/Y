pragma circom 2.0.0;
// The control for `multi_var_decl.circom`: a real `{ }` block DOES open a
// scope, in circom (error[T2021] on the leak) as in Y. Making `var a, b;`
// work by removing block scoping altogether would break this.
template M() {
    signal input i;
    signal output o;
    { var hidden = 99; }
    o <== hidden * i;
}
component main = M();
