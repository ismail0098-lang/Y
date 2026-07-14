fn main(x: Field, y: Field) -> pub Field {
    let mut temp = x;
    for _i in 0..1000000 {
        temp = temp * y;
    }
    temp
}
