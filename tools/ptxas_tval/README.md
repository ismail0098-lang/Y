# `ptxas` translation validator

Symbolically execute a kernel's PTX and the SASS `ptxas` produced from that exact
file; ask an SMT solver whether the two can ever store different values.

**An opcode or operand form neither executor models is a HARD ERROR, never a
guess.** A symbolic executor that guesses at an instruction it does not know
produces a proof about a program nobody wrote.

The write-up — method, results, findings, what is *not* claimed — is
[`docs/ptxas_translation_validation.md`](../../docs/ptxas_translation_validation.md).

## Requirements

`python3` with `z3-solver`; `ptxas` and `nvdisasm` from the CUDA toolkit.
Nothing here is built or run by `cargo test`: it needs a GPU toolchain and takes
minutes to hours per kernel.

## Run

```sh
./build_corpus.sh    # tests/*.ptx -> corpus/ and o1/, via ptxas + nvdisasm
./regress.sh         # the standing straight-line results, ~35 s
python3 loopval.py o1/exact_pv.ptx o1/exact_pv.sass 60 wide
python3 smemval.py smut/smem_roundtrip.ptx smut/smem_roundtrip.sass 60 wide
```

`corpus/` and `o1/` are generated and not committed: a `.cubin` is a machine-specific ELF
and a `.sass` is a disassembly of one. All 66 rebuild byte-identically to the
ones the published results were measured on.

## Layout

| | |
|---|---|
| `ptxexec.py` `sassexec.py` | the two symbolic executors |
| `smem.py` | shared memory as a z3 array; barriers as an uninterpreted `H_k` |
| `fpmode.py` | float macro-op table, with a `validated` flag per identification |
| `mulmode.py` `conc.py` | the multiplier ladder (`uf` / `wide` / `direct`) and concretisation |
| `batch.py` | the obligations, and `same_if` — guard-relative address matching |
| `tval.py` `loopval.py` `smemval.py` | drivers: straight-line, loop, shared memory |
| `cfg.py` `loopcfg.py` `params.py` | control-flow and signature parsing |
| `scope2.py` `depth.py` `tractable.py` `smemdepth.py` `barregion.py` | measurement |
| `gap.py` | the dynamic gap — what the executor genuinely refuses, not its first refusal; `--rank` adds cost **and reach** |
| `loopgap.py` | why `loopval` refuses each kernel that has a loop — the structural gate `gap.py` cannot see |
| `cbank_abi.py` `cbank_abi.c` | referee the const-bank ABI against `ptxas` **and** the device |
| `fpclass.py` `expand.py` `contract.py` `unroll.py` `olevel.py` `muls.py` | measurement |
| `gmut.sh` `lmut.sh` `smut.sh` `rmut.sh` + `muts/` | mutation tables; **the control row is first** |
| `mkbase.sh` `restore.sh` | baseline snapshot/restore for the mutation harnesses |
| `fma/ div/ loop/ smut/ synth/` | small hand-built fixtures, including the negative controls and `synth/nostore` (a kernel that stores nothing) |

## The row that makes the table mean something

`fma/plain` and `fma/rn` are the same kernel; `plain` lets `ptxas` contract
`mul.f32`+`add.f32` into one `FFMA`, and the validator answers `sat` with a
counterexample. A validator that always says VALIDATED would report every other
row identically. Keep that control passing — i.e. failing.
