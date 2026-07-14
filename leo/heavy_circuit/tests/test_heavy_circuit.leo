// The 'test_heavy_circuit' test program.
import heavy_circuit.aleo;
program test_heavy_circuit.aleo {
    @test
    @should_fail
    fn test_main_fails() {
        let result: u32 = heavy_circuit.aleo::main(2u32, 3u32);
        assert_eq(result, 3u32);
    }

    @noupgrade
    constructor() {}
}
