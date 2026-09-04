"""Does the .rn repair unlock any kernel TODAY?

The nine CONTRACTION-only kernels are the entire set option (a) can help.
Cross them against what else blocks them.
"""
import cfg,re,os
NINE=['gemm_f16_bias_relu_64','gemm_f16_bias_relu_512','gemm_f16_bias_relu_1024',
      'gemm_f16_bias_relu_2048','gemm_f16_bias_relu_4096','gemm_f16_bias_relu_8192',
      'int8_gemm_scaled','naive_gemm_f32','y_cpu_matmul']
print(f'{"kernel (contraction-only)":34s}{"insn":>6}{"loops":>7}{"shmem":>7}   other blocker')
unlocked=0
for k in NINE:
    n,fwd,back,rec,mem=cfg.analyse(f'corpus/{k}.sass')
    P=open(f'corpus/{k}.ptx').read()
    blk=[]
    if back: blk.append(f'{back} back edge(s) -> needs loop invariants')
    if mem:  blk.append(f'{mem} shared-mem/atomic op(s)')
    if not blk: unlocked+=1; blk=['NONE -- would be unlocked']
    print(f'{k:34s}{n:6d}{back:7d}{mem:7d}   {"; ".join(blk)}')
print(f'\nkernels the .rn repair unlocks today: {unlocked} / {len(NINE)}')
