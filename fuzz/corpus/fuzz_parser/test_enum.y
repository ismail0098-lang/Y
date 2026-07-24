enum Expr { A(I32), B } fn test(e: Expr) { if e.tag == Expr_TAG_A { let val: I32 = e.data.A._0; } }
