# Compiler regression verification — 2026-09-13

**27 regression tests fail on the original revision and pass on the patched compiler. Four compatibility controls pass on both. All 62 test invocations completed; none was ignored or skipped.**

The original revision is `8b7122ab6cb78c5224e9d22c2d10aa41ca9dab50`. It was extracted with `git archive` into a separate directory. The same seven test files, checked by SHA-256, were compiled and run against both versions. Both Rust builds succeeded. Compiler input hashes were checked before and after the run to detect changes during verification.

| Reviewed issue | Original fails → patched passes | Controls pass on both | Observed original failure |
| --- | ---: | ---: | --- |
| 1. Compile-time assertions | 3 | 1 | False and unproved assertions emit artifacts; the C API returns success. |
| 2. Exact LLVM integers | 3 | 0 | Generated code fails independent C integer expectations, including 9007199254740993. |
| 3. Bounds and integer widths | 4 | 0 | Invalid branch/compound/match ranges are accepted; valid widened arithmetic is rejected. |
| 4. Frontend semantics | 3 | 1 | Undefined names, missing arguments, and numeric conditions receive no diagnostic. |
| 5. Numeric literals | 6 | 2 | Oversized or malformed values become zero, signed minimum is corrupted, and nonfinite floats are accepted. |
| 6. LLVM/PTX state isolation | 6 | 0 | Unrelated definitions affect diagnostics, signed results, scratch registers, or labels. |
| 7. ZK field context | 2 | 0 | A retained value no longer decodes as 7 and retained witness results change after switching fields. |

The LLVM integer tests execute emitted code against independent C expectations. Original execution fails at `-O0`; patched execution completes at both `-O0` and `-O2`. The LLVM signedness test also demonstrates an incorrect runtime value on the original revision.

The two PTX counter tests are structural checks. Their original modules assemble successfully, then fail isolation comparisons: the second kernel starts double scratch registers at `%fd8` instead of `%fd0`, or uses `$IF_END_3` instead of `$IF_END_1`. The double fixture has no branches; the label fixture allocates no double registers. These tests establish state isolation, not GPU numerical correctness.

The integer-width test fails originally on rejection of valid I64 widening; its narrowing example is already rejected by the original compiler. The match-arm test constructs an AST because the source parser cannot express those block-valued arms. Each multi-assertion test stops at its first original failure, so the results count discriminating test functions rather than claiming every subcase independently fails originally. Expected, caught mixed-field panics in patched ZK logs test rejection behavior.

Two test files were adapted to APIs present in both revisions, preserving behavioral assertions. LLVM execution was moved before structural IR checks. The original combined PTX counter test did not allocate double registers; it was replaced with separate fixtures that demonstrably exercise each counter. No production compiler code changed during this verification turn.

Reproduce from the repository root:

```sh
python3 tools/verify_review_regressions.py
```

The runner requires Cargo, clang, ptxas, and Z3, and uses offline locked dependencies. Set `Y_Z3_PATH` if Z3 is outside the repository venv and PATH. It retains original sources, independent build outputs, and per-test logs under the printed `/tmp/y-regression-audit-*` directory. It requires all four named controls to pass on both revisions and every other discovered regression to fail originally and pass after the fix. Missing tools, build failures, skipped tests, and unexpected transitions fail the audit.

[Machine-readable evidence with all 62 complete test logs, build logs, tool versions, and source/test hashes](compiler_regressions_2026-09-13.json). The complete working artifacts for this run remain in `/tmp/y-review-before-after-final-20260913`.

Tool versions: Cargo/rustc 1.97.1, clang 22.1.8, ptxas 13.3.73, Z3 5.0.0.

Compiler input manifest SHA-256: `c267d425990ca813dc0887bf34606ff39e084dc281186a69ed86d953f9fe5a1f`.

| Test | Original | Patched |
| --- | --- | --- |
| `compile_time_assertions::embedded_cpu_entrypoint_propagates_frontend_assertion_errors` | fail | pass |
| `compile_time_assertions::false_assertions_fail_before_emission_at_both_syntax_sites` | fail | pass |
| `compile_time_assertions::true_assertions_are_evaluated_and_erased` | pass | pass |
| `compile_time_assertions::unsupported_nonboolean_and_invalid_arithmetic_are_not_verified` | fail | pass |
| `bounds_control_flow::annotated_integer_width_controls_the_stored_range` | fail | pass |
| `bounds_control_flow::compound_assignment_updates_or_invalidates_the_range` | fail | pass |
| `bounds_control_flow::each_branch_starts_from_the_entry_facts_and_exits_are_joined` | fail | pass |
| `bounds_control_flow::match_arms_do_not_overwrite_each_others_facts` | fail | pass |
| `frontend_semantics::calls_validate_arity_argument_types_and_return_types` | fail | pass |
| `frontend_semantics::operators_have_types_and_boolean_conditions_are_required` | fail | pass |
| `frontend_semantics::undeclared_names_are_not_unknown_typed_values` | fail | pass |
| `frontend_semantics::valid_forward_calls_polymorphic_literals_and_struct_fields_work` | pass | pass |
| `integer_literal_diagnostics::integer_errors_name_the_literal_and_supported_range_in_every_context` | fail | pass |
| `integer_literal_diagnostics::integer_overflow_preserves_its_spelling_and_is_refused` | fail | pass |
| `integer_literal_diagnostics::malformed_float_is_refused_instead_of_becoming_zero` | fail | pass |
| `integer_literal_diagnostics::nonfinite_float_is_refused` | fail | pass |
| `integer_literal_diagnostics::ordinary_decimal_floats_and_range_tokens_still_work` | pass | pass |
| `integer_literal_diagnostics::representable_nonnegative_integers_are_preserved` | pass | pass |
| `integer_literal_diagnostics::signed_minimum_is_preserved_including_leading_zeroes` | fail | pass |
| `integer_literal_diagnostics::the_cli_refuses_a_u64_literal_it_cannot_lower_without_writing_ir` | fail | pass |
| `llvm_zero_drift_integer::every_accumulation_spelling_preserves_wide_terms_and_readers` | fail | pass |
| `llvm_zero_drift_integer::initialization_and_readback_preserve_the_full_signed_integer` | fail | pass |
| `llvm_zero_drift_integer::integer_terms_keep_signedness_when_widened` | fail | pass |
| `backend_function_state::llvm_buffer_element_facts_do_not_escape_their_function` | fail | pass |
| `backend_function_state::llvm_directives_are_local_to_functions_and_kernels` | fail | pass |
| `backend_function_state::llvm_inferred_locals_do_not_inherit_old_signedness` | fail | pass |
| `backend_function_state::ptx_directives_are_local_to_each_kernel` | fail | pass |
| `backend_function_state::ptx_double_registers_start_fresh_in_each_kernel` | fail | pass |
| `backend_function_state::ptx_labels_start_fresh_in_each_kernel` | fail | pass |
| `zk_field_context::retained_circuits_and_witnesses_survive_interleaved_fields` | fail | pass |
| `zk_field_context::retained_elements_keep_their_value_arithmetic_and_identity` | fail | pass |

This closes the before/after regression gap for the seven reported issues. It is evidence for those fixes, not a proof of the entire compiler.
