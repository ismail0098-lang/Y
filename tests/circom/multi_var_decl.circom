pragma circom 2.0.0;
// `var a, b;` declares BOTH in the enclosing scope. Y parsed this into a
// `Stmt::Block`, which opens a scope and pops it again, so both names were
// gone by the next statement -- "`a` is not a variable in scope".
template M() {
    signal input i;
    signal output o;
    var a, b;
    a = 5;
    b = 7;
    o <== (a + b) * i;
}
component main = M();
