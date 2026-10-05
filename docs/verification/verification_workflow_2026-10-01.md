# Verification workflow review — 2026-10-01

This review examined SMT invariant checking, Coq proof gates, PTX/SASS
translation validation, retained cubin construction/loading, and the compiler
regression audit runner. The repository already contained substantial tracked
and untracked changes; they were retained.

## Findings and fixes

| Area | Reproduced error | Change and regression coverage |
| --- | --- | --- |
| SMT aliases | A stored `&mut x` passed to a callee bypassed the syntactic reference check. Y verified `x >= 0` although the compiled program set `x` to `-1`. Calls and indirect writes before loop entry also retained stale initializer ranges. | Inspect reference types, including struct and enum payloads; clear aliased range facts after calls and indirect writes. `tests/smt_reference_aliases.rs` covers calls inside/before loops, branches, assignment/initializer calls, aggregates, dotted calls, indirect stores, and valid scalar controls. |
| Shared float loads | Scalar/vector `ld.shared.f32` wrote the integer register file, retaining a stale float. Incorrect SASS could validate while correct SASS failed. | Route float loads to float registers and refuse unsupported destination register files. Positive and negative tests in `tests/translation_validator_soundness.py`. |
| Parameter ABI | An array parameter was parsed as a scalar or skipped, shifting later constant-bank arguments. Incorrect SASS using the wrong output-pointer offset could validate. | Parse complete declarations and refuse unsupported arrays, explicit alignment, subword packing, duplicate names, and parameter load forms. |
| Shared validation entry points | Direct `tval` and `batch` callers omitted shared-memory equality, alignment, and barrier obligations that the `smemval` wrapper checked. | Use one shared-effect checker from both direct paths; `smemval` delegates to `batch`. Dropped writes/barriers and misalignment regressions exercise the direct APIs. |
| Proof content gates | Commented-out `Print Assumptions` commands and identifier prefixes could satisfy a required content control. A commented-out required example still compiled. | Match active Coq tokens and complete names; ignore comments/strings when counting assumption commands. `tests/proofs_are_checked.rs` includes parser regressions and compiles every proof. |
| Artifact construction | Valid target comments caused refusal; successful publication could overwrite a newly created empty destination. | Normalize comments on the input snapshot and reserve a new destination exclusively. Failed builds remove only empty reservations and preserve foreign contents. |
| Validator process lifetime | Shared validation before `exact_pv` in the same Python process changed `VALIDATED`/14 obligations to `UNPROVED`/16 because of an existing Z3 node-order-sensitive multiplication abstraction. | Run artifact validation in a fresh process using the selected interpreter, with a checked result schema and a finite deadline. Regression coverage builds consecutive real artifacts after the triggering preamble. Direct in-process validator APIs retain this existing completeness limitation. |
| Python tool discovery | Cargo's artifact gate used system Python and skipped real integration despite a repository environment containing Z3. | Share interpreter discovery between artifact, translation soundness, and exact-PV device gates. Preserve `Y_TVAL_PYTHON` override and explicit missing-tool notices. |
| Audit evidence | The audit hashed its runner and hardware profile after execution, permitting evidence to describe inputs different from those tested. | Snapshot hashes before preparation and reject changes in the runner or either checkout's hardware profile. `tests/verification_workflow_regressions.py` uses controlled mutations and a stable control. |

## Before/after evidence

The same alias regression suite was linked against a saved copy of the
pre-review type checker: six negative tests failed because invalid invariants
were accepted; the valid scalar control passed. All seven tests pass with the
corrected checker. The executable saved-reference reproducer returned `-1`
after the original compiler accepted the `x >= 0` invariant.

Commenting out `ZkControlFlow.v`'s required `low_tail_is_wrong_when_nested`
example passed the original content gate and Coq, but fails the corrected
content gate. Hardware-profile and runner mutations likewise pass the original
audit and fail the corrected audit; stable inputs retain their original hashes.

## Validation and limits

Targeted SMT, reference-type, scalar-rule, Coq, translation soundness, artifact,
and audit workflow checks were run with the available external tools. Real
artifact tests invoke `ptxas`, `nvdisasm`, and Python Z3, verify the retained
image, and reject a modified real cubin. GPU execution tests cannot exercise
the device in this environment because the CUDA driver is unavailable.

The workspace sweep includes the root package, `y-gpu`, and the `zk` feature.
It exposes existing failing kernel fixtures as well as optional-tool/device
skips. Representative failures were reproduced with the saved pre-review
checker: GEMM kernels have unconstrained signed dimensions, while PTX's
unsigned `for` comparison requires a proved nonnegative end; the grid-stride
alias fixtures lack a bound excluding induction-variable overflow, and the
generated MSM fixture reuses a name rejected by the name-based SMT model. These
failures were retained instead of weakening the arithmetic proof obligations.

Clearing aliased ranges is conservative: without an alias/effect model, calls
receiving references may require callers to reestablish range facts. Declared
`@bounds` and inferred ranges currently share storage, which limits recovery
of facts for explicitly bounded variables after invalidation.

Reproduce the focused checks from the repository root:

```bash
Y_Z3_PATH="$PWD/venv/bin/z3" cargo test --offline --features zk \
  --test smt_reference_aliases --test smt_machine_arithmetic \
  --test safe_invariant_enforcement --test reference_types_are_checked \
  --test type_checker_scalar_rules --test proofs_are_checked \
  --test translation_validator_soundness --test exact_pv_artifact_binding \
  --test verification_workflow_regressions -- --nocapture

Y_Z3_PATH="$PWD/venv/bin/z3" cargo test --offline --workspace \
  --features zk --tests --no-fail-fast
```

Local review logs are under `/tmp/y-verification-*`; they are session artifacts,
not committed evidence. Final results:

| Check | Result |
| --- | --- |
| SMT/reference/scalar focused targets | 57 Rust tests passed, including all 7 new alias tests |
| Coq gate | 8 gate tests passed; all 24 proof files compiled |
| Artifact gate | 31 Python tests passed, including real toolchain and repeated-build tests; no skips |
| Translation soundness gate | 19 Python tests passed; no skips |
| Audit runner gate | 4 Python tests passed through its Rust wrapper |
| Full workspace with `zk` | 177 targets: 1,100 reported passing, 59 failed, 50 ignored; exit 101 |
| Whitespace validation | `git diff --check` passed |

Some reported passes in the workspace are runtime skips, especially device
checks without a CUDA driver. They are not evidence of GPU execution. The
sampled remaining failures were independently reproduced with the saved
pre-review checker; the full original workspace was not rebuilt for this
comparison.

The 19 failing targets were:

- `certificate_states_its_trust_boundary`
- `committed_ptx_artifacts`
- `emitted_attribute_groups`
- `exact_gemm_allocation_failure`
- `exact_gemm_certificate`
- `exact_gemm_msplit`
- `exact_gemm_packing_model`
- `exact_gemm_requires_its_hardware`
- `exact_gemm_spawn_failure`
- `exact_gemm_thread_invariance`
- `exact_gemm_thread_sanitizer`
- `exact_gemm_tiling_model`
- `exact_pv_proof`
- `gemm_substitution_differential`
- `host_cache_directives`
- `let_binding_aliasing`
- `manual_examples`
- `source_surface`
- `zero_drift_backend_agreement`

Detailed final logs: `/tmp/y-verification-workspace-final.log`,
`/tmp/y-verification-smt-final.log`, `/tmp/y-verification-artifacts-final.log`,
and `/tmp/y-verification-workflow.log`. The unchanged-checker comparisons are
in `/tmp/y-verification-alias-before.log` and
`/tmp/y-verification-pre-session-checks/`.

