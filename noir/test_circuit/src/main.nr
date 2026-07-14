fn main(x: Field, y: Field) -> pub Field {
    let product = x * y;
    let mut loop_sum = 0;
    for i in 0..5 {
        loop_sum = loop_sum + x * (i as Field);
    }
    let mut cond_val = 0;
    if x == y {
        cond_val = 100;
    } else {
        cond_val = 200;
    }
    product + loop_sum + cond_val
}
