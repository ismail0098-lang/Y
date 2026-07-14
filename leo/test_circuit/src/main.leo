// The 'test_circuit' program.
program test_circuit.aleo {
    @noupgrade
    constructor() {}

    fn main(public x: u32, public y: u32) -> u32 {
        let product: u32 = x * y;
        let loop_sum: u32 = 0u32;
        for i: u32 in 0u32..5u32 {
            loop_sum = loop_sum + x * i;
        }
        let cond_val: u32 = 0u32;
        if x == y {
            cond_val = 100u32;
        } else {
            cond_val = 200u32;
        }
        return product + loop_sum + cond_val;
    }
}
