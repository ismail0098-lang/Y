// The 'dot_product' program.
program dot_product.aleo {
    @noupgrade
    constructor() {}

    fn main(public x: u32, public y: u32) -> u32 {
        let sum: u32 = 0u32;
        let a: u32 = x;
        let b: u32 = y;
        for i: u32 in 0u32..100000u32 {
            a = a + 1u32;
            b = b + 1u32;
            sum = sum + a * b;
        }
        return sum;
    }
}
