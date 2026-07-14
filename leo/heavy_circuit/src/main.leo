// The 'heavy_circuit' program.
program heavy_circuit.aleo {
    @noupgrade
    constructor() {}

    fn main(public x: u32, public y: u32) -> u32 {
        let temp: u32 = x;
        for i: u32 in 0u32..1000000u32 {
            temp = temp * y;
        }
        return temp;
    }
}
