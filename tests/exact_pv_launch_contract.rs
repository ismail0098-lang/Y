//! CPU-only contract checks: complete small launch enumeration and arithmetic
//! boundary controls. Driver-spy tests in cuda_runtime exercise ABI/ownership.

use y::verified_exact_pv::ExactPvShape;

#[test]
fn every_checked_launch_coordinate_has_exactly_its_source_indices() {
    for bmax in 1..=3 {
        for qmax in 1..=3 {
            for tmax in 1..=4 {
                for dmax in 1..=5 {
                    let shape = ExactPvShape::new(bmax, qmax, tmax, dmax).unwrap();
                    let [np, nv, no] = shape.element_counts();
                    assert_eq!(
                        shape.required_bytes(),
                        [np as usize * 4, nv as usize, no as usize * 8]
                    );
                    assert_eq!(shape.grid(), (qmax as u32, bmax as u32, 1));
                    assert_eq!(shape.block(), (dmax as u32, 1, 1));
                    let mut ownership = vec![0; no as usize];
                    for b in 0..bmax {
                        for q in 0..qmax {
                            for d in 0..dmax {
                                let oi = (b * qmax + q) * dmax + d;
                                assert!((0..no as i64).contains(&oi));
                                ownership[oi as usize] += 1;
                                for t in 0..tmax {
                                    let pi = (b * qmax + q) * tmax + t;
                                    let vi = (b * tmax + t) * dmax + d;
                                    assert!((0..np as i64).contains(&pi));
                                    assert!((0..nv as i64).contains(&vi));
                                    assert!(pi <= i32::MAX as i64 && vi <= i32::MAX as i64);
                                }
                            }
                        }
                    }
                    assert!(ownership.iter().all(|&count| count == 1));
                }
            }
        }
    }
}

#[test]
fn signed_empty_ranges_are_refused_without_claiming_the_positive_t_theorem() {
    for dimension in 0..4 {
        for value in [i64::MIN, -1, 0, i32::MAX as i64 + 1, i64::MAX] {
            let mut dims = [1; 4];
            dims[dimension] = value;
            assert!(ExactPvShape::new(dims[0], dims[1], dims[2], dims[3]).is_err());
        }
    }
}

#[test]
fn index_lengths_and_full_operand_domain_accumulator_bound_are_exact() {
    let limit = ((i64::MAX as u128) / (u32::MAX as u128 * 128)) as i64;
    assert_eq!(limit, 16_777_216);
    assert!(ExactPvShape::new(1, 1, limit, 1).is_ok());
    assert!(ExactPvShape::new(1, 1, limit + 1, 1).is_err());
    // Positive dimensions individually fit; the element lengths do not.
    assert!(ExactPvShape::new(
        i32::MAX as i64,
        i32::MAX as i64,
        i32::MAX as i64,
        i32::MAX as i64
    )
    .is_err());
    for dims in [
        [1, i32::MAX as i64, 2, 1],
        [2, 1, 1, i32::MAX as i64],
        [1, i32::MAX as i64, 1, 2],
    ] {
        assert!(ExactPvShape::new(dims[0], dims[1], dims[2], dims[3]).is_err());
    }
    #[cfg(target_pointer_width = "64")]
    {
        let edge = ExactPvShape::new(1, i32::MAX as i64, 1, 1).unwrap();
        assert_eq!(edge.element_counts(), [i32::MAX, 1, i32::MAX]);
    }
}
