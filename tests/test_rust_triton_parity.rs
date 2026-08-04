use y::autotuner::{Autotuner, Precision};
use y::ptx_emitter::PtxEmitter;
use y::sentinel::HardwareProfile;
use y::ast::{Program, Item, KernelDecl, Block, Stmt, Expr, Span};
use y::c_api::{y_autotune_select_config_json, y_block_cdiv, y_free_string};
use std::ffi::CStr;

#[test]
fn test_rust_autotuner_cache() {
    let hw = HardwareProfile::default();
    let config1 = Autotuner::autotune(1024, 1024, 1024, &hw, Precision::F16);
    let config2 = Autotuner::autotune(1024, 1024, 1024, &hw, Precision::F16);
    assert_eq!(config1.cta_m, config2.cta_m);
    assert_eq!(config1.num_warps, config2.num_warps);
}

#[test]
fn test_rust_ptx_block_cdiv_and_arange_emission() {
    let mut emitter = PtxEmitter::new();
    let hw = HardwareProfile::default();

    let program = Program {
        items: vec![
            Item::Kernel(KernelDecl {
                requires: vec![],
                name: "test_kernel".to_string(),
                params: vec![],
                body: Block {
                    stmts: vec![
                        Stmt::Expr(Expr::Call {
                            func: Box::new(Expr::Ident("block_cdiv".to_string(), Span { line: 1, col: 1 })),
                            args: vec![
                                Expr::IntLit(100, Span { line: 1, col: 1 }),
                                Expr::IntLit(32, Span { line: 1, col: 1 }),
                            ],
                            span: Span { line: 1, col: 1 },
                        }),
                        Stmt::Expr(Expr::Call {
                            func: Box::new(Expr::Ident("block_arange".to_string(), Span { line: 1, col: 1 })),
                            args: vec![],
                            span: Span { line: 1, col: 1 },
                        }),
                    ],
                    span: Span { line: 1, col: 1 },
                },
                tile: None,
                span: Span { line: 1, col: 1 },
            })
        ],
    };

    let ptx_output = emitter.emit_program(&program, &hw);
    assert!(ptx_output.contains("Y BLOCK CDIV - CEIL DIVISION"));
    assert!(ptx_output.contains("div.s32"));
    assert!(ptx_output.contains("Y BLOCK ARANGE - 1D INDEX GENERATOR"));
    assert!(ptx_output.contains("%tid.x"));
}


#[test]
fn test_rust_c_api_exports() {
    assert_eq!(y_block_cdiv(10, 3), 4);
    assert_eq!(y_block_cdiv(100, 32), 4);

    unsafe {
        let json_ptr = y_autotune_select_config_json(1024, 1024, 1024, false);
        assert!(!json_ptr.is_null());
        let json_str = CStr::from_ptr(json_ptr).to_str().unwrap();
        assert!(json_str.contains("cta_m"));
        assert!(json_str.contains("num_warps"));
        y_free_string(json_ptr);
    }
}


