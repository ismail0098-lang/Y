//! Thin `Result`-returning wrappers over the CUDA driver bindings.

use y::cuda_runtime::{CudaContext, DeviceBuffer, KernelModule};

use crate::error::{Error, Result};

pub(crate) fn alloc(ctx: &CudaContext, bytes: usize) -> Result<DeviceBuffer> {
    ctx.alloc(bytes.max(1)).map_err(|_| Error::Alloc { bytes })
}

pub(crate) fn upload_u32(ctx: &CudaContext, v: &[u32]) -> Result<DeviceBuffer> {
    let d = alloc(ctx, v.len().max(1) * 4)?;
    write_u32(ctx, &d, v)?;
    Ok(d)
}

pub(crate) fn write_u32(ctx: &CudaContext, d: &DeviceBuffer, v: &[u32]) -> Result<()> {
    let bytes = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
    ctx.memcpy_htod_at(d, 0, bytes)
        .map_err(|e| Error::Cuda(e.to_string()))
}

pub(crate) fn read_u32(ctx: &CudaContext, d: &DeviceBuffer, len: usize) -> Result<Vec<u32>> {
    let mut raw = vec![0u8; len * 4];
    ctx.memcpy_dtoh_at(&mut raw, d, 0)
        .map_err(|e| Error::Cuda(e.to_string()))?;
    Ok(raw
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

pub(crate) fn launch(
    ctx: &CudaContext,
    m: &KernelModule,
    threads: usize,
    args: &[u64],
) -> Result<()> {
    if threads == 0 {
        return Ok(());
    }
    let block = 256u32.min(threads as u32).max(1);
    let grid = (threads as u32).div_ceil(block);
    ctx.launch(m, (grid, 1, 1), (block, 1, 1), 0, args)
        .map_err(|e| Error::Cuda(e.to_string()))
}

pub(crate) fn sync(ctx: &CudaContext) -> Result<()> {
    ctx.synchronize().map_err(|e| Error::Cuda(e.to_string()))
}
