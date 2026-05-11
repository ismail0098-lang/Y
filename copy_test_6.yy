struct MyStruct {
    a: i64,
    b: i64,
}

fn MyStruct_new(a: i64, b: i64) -> MyStruct {
    return MyStruct { a: a, b: b };
}

fn test(s: &MyStruct) {
    print_int(s.a);
}

@unsafe
fn main() -> i32 {
    let s = MyStruct_new(42, 100);
    test(&s);
    return 0;
}
