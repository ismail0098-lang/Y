fn main(x: Field, y: Field) -> pub Field {
    let mut sum = 0;
    let mut a = x;
    let mut b = y;
    for _i in 0..100000 {
        a = a + 1;
        b = b + 1;
        sum = sum + a * b;
    }
    sum
}
