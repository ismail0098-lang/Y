@require(sm >= 89)
kernel drift_demo() {
    // Q32.32 is natively supported by our simulated GPU profile
    @ZeroDrift
    let acc: Q32.32 = Fragment::zero();

    // F16 is not in the drift_free_types list, so it will trigger a warning
    @ZeroDrift
    let fast_acc: F16 = Fragment::zero();

    // Test heuristic branching vs predication
    if 1 {
        let x: u32 = 10;
        let y: u32 = 20;
    } else {
        let z: u32 = 30;
    }

    // Test IMAD.WIDE fast path
    let ptr_math: u32 = acc * fast_acc;

    // Test Barrier Hoisting
    barrier::sync();
    let hoist1: u32 = 10 + 20;
    let hoist2: u32 = 30 * 40;
}
