// hello.y -- Y-Lang test program for C backend validation

fn add(a: I32, b: I32) -> I32 {
    return a + b;
}

fn factorial(n: I32) -> I32 {
    if n <= 1 {
        return 1;
    }
    return n * factorial(n - 1);
}

fn fizzbuzz(n: I32) {
    let i: I32 = 1;
    while i <= n {
        if i % 15 == 0 {
            println("FizzBuzz");
        } else if i % 3 == 0 {
            println("Fizz");
        } else if i % 5 == 0 {
            println("Buzz");
        } else {
            print_int(i);
        }
        i += 1;
    }
}

fn main() {
    let result: I32 = add(3, 4);
    print_int(result);
    let fact: I32 = factorial(10);
    print_int(fact);
    fizzbuzz(20);
    return;
}
