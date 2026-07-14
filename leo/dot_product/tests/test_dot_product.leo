// The 'test_dot_product' test program.
import dot_product.aleo;
program test_dot_product.aleo {
    @test
    @should_fail
    fn test_main_fails() {
        let result: u32 = dot_product.aleo::main(2u32, 3u32);
        assert_eq(result, 3u32);
    }

    @noupgrade
    constructor() {}
}
