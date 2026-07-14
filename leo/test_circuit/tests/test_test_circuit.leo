// The 'test_test_circuit' test program.
import test_circuit.aleo;
program test_test_circuit.aleo {
    @test
    @should_fail
    fn test_main_fails() {
        let result: u32 = test_circuit.aleo::main(2u32, 3u32);
        assert_eq(result, 3u32);
    }

    @noupgrade
    constructor() {}
}
