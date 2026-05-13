@unsafe
fn fibonacci(n: I32) -> I32 {
    if n <= 1 {
        return n;
    }
    return fibonacci(n - 1) + fibonacci(n - 2);
}

@unsafe
fn main() {
    println("--- Y-Lang Execution Test ---");
    println("Calculating Fibonacci of 10...");
    
    let result: I32 = fibonacci(10);
    
    println("Result:");
    print_int(result);
    
    println("");
    println("Y-Lang to Native LLVM compilation is fully functional!");
    
    return;
}
