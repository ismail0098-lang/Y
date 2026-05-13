struct MyStruct {
    a: i64,
    b: &String,
}

fn MyStruct_new(a: i64, b: &String) -> MyStruct {
    return MyStruct { a: a, b: b };
}

@unsafe
fn main() -> i32 {
    let s_str = String_new("hello");
    let s = MyStruct_new(42, &s_str);
    return 0;
}
