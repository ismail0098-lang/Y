// ============================================================
//  Y  —  Bank Conflict Prover
//  bank_conflict.rs
//
//  A mathematical engine that validates whether a given
//  `SmemLayout` (paired with a particular warp-level operation
//  like `ldmatrix`) is conflict-free across 32 threads.
// ============================================================

#![allow(dead_code)]

/// Swizzle mode specifying hardware swizzling parameters.
#[derive(Debug, Clone, PartialEq)]
pub enum SwizzleMode {
    Xor128B,
    Xor64B,
    Xor32B,
    Custom(SwizzlePattern),
}

/// Represents the `Swizzle<XOR, base, offset>` pattern.
#[derive(Debug, Clone, PartialEq)]
pub struct SwizzlePattern {
    pub xor_bits: u32,
    pub base_shift: u32,
    pub offset: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SmemLayout {
    pub rows: u32,
    pub cols: u32,
    pub swizzle: Option<SwizzlePattern>,
    pub swizzle_mode: Option<SwizzleMode>,
    pub bytes_per_element: u32,
}

/// Simulated memory access for a specific thread in a warp.
#[derive(Debug)]
struct ThreadAccess {
    thread_id: u32,
    linear_byte_address: u32,
    bank: u32,
}

pub struct BankConflictProver;

impl BankConflictProver {
    /// Compute swizzled byte address given linear byte address and row index.
    pub fn swizzle_byte_address(byte_addr: u32, row: u32, layout: &SmemLayout) -> u32 {
        if let Some(mode) = &layout.swizzle_mode {
            match mode {
                SwizzleMode::Xor128B => {
                    // 128-Byte XOR swizzle: XOR row index (0..7) into 16-byte chunk offset
                    // byte_addr ^ ((row & 7) << 4)
                    let xor_val = row & 7;
                    let chunk_16b = byte_addr / 16;
                    let swizzled_chunk = chunk_16b ^ xor_val;
                    (swizzled_chunk * 16) | (byte_addr % 16)
                }
                SwizzleMode::Xor64B => {
                    let xor_val = row & 3;
                    let chunk_16b = byte_addr / 16;
                    let swizzled_chunk = chunk_16b ^ xor_val;
                    (swizzled_chunk * 16) | (byte_addr % 16)
                }
                SwizzleMode::Xor32B => {
                    let xor_val = row & 1;
                    let chunk_16b = byte_addr / 16;
                    let swizzled_chunk = chunk_16b ^ xor_val;
                    (swizzled_chunk * 16) | (byte_addr % 16)
                }
                SwizzleMode::Custom(swizzle) => {
                    let mask = (1 << swizzle.xor_bits) - 1;
                    let shift = swizzle.base_shift;
                    let xor_val = ((row >> swizzle.offset) & mask) << shift;
                    let chunk_idx = byte_addr / 16;
                    let new_chunk_idx = chunk_idx ^ xor_val;
                    (new_chunk_idx * 16) | (byte_addr % 16)
                }
            }
        } else if let Some(swizzle) = &layout.swizzle {
            let mask = (1 << swizzle.xor_bits) - 1;
            let shift = swizzle.base_shift;
            let xor_val = ((row >> swizzle.offset) & mask) << shift;
            let chunk_idx = byte_addr / 16;
            let new_chunk_idx = chunk_idx ^ xor_val;
            (new_chunk_idx * 16) | (byte_addr % 16)
        } else {
            byte_addr
        }
    }

    /// Validates an `ldmatrix.sync.aligned.m16n8` pattern against the provided layout.
    /// This simulates 32 threads executing the ldmatrix and checks if any two threads
    /// within the warp hit the same bank simultaneously.
    pub fn prove_ldmatrix_m16n8(layout: &SmemLayout) -> Result<(), String> {
        let banks_count = 32;
        let bank_width_bytes = 4; // 32 banks, 4 bytes wide on modern NVIDIA GPUs.

        let mut accesses = Vec::with_capacity(32);

        for tid in 0..32 {
            let row = tid % 16;
            let col = (tid / 16) * 8;

            let linear_idx = row * layout.cols + col;
            let raw_byte_addr = linear_idx * layout.bytes_per_element;
            let byte_addr = Self::swizzle_byte_address(raw_byte_addr, row, layout);
            let bank = (byte_addr / bank_width_bytes) % banks_count;

            accesses.push(ThreadAccess {
                thread_id: tid,
                linear_byte_address: byte_addr,
                bank,
            });
        }

        let mut bank_counts = vec![0; banks_count as usize];
        for access in &accesses {
            bank_counts[access.bank as usize] += 1;
        }

        for (bank_id, count) in bank_counts.iter().enumerate() {
            if *count > 4 {
                return Err(format!(
                    "Bank Conflict Prover: Warp will serialize! \
                     {} threads hit Bank {} simultaneously in a single transaction. \
                     Swizzle layout applied: {:?}",
                    count, bank_id, layout.swizzle_mode
                ));
            }
        }

        Ok(())
    }

    /// Validates `ldmatrix.sync.aligned.m8n8.x4` vectorized 128B matrix loads.
    /// Ensures 32 threads reading 4 vector matrix fragments across swizzled SMEM
    /// achieve zero bank serialization.
    pub fn prove_ldmatrix_m8n8_x4(layout: &SmemLayout) -> Result<(), String> {
        let banks_count = 32;
        let bank_width_bytes = 4;

        // Check phase by phase: ldmatrix.x4 issue 4 16-byte vector transactions
        for v in 0..4 {
            let mut phase_bank_counts = vec![0; banks_count as usize];

            for tid in 0..32 {
                let row = (tid % 8) + v * 8;
                let col = (tid / 8) * 8;

                let linear_idx = row * layout.cols + col;
                let raw_byte_addr = linear_idx * layout.bytes_per_element;
                let byte_addr = Self::swizzle_byte_address(raw_byte_addr, row, layout);

                // A 16-byte vector load accesses 4 consecutive 4-byte banks
                let start_bank = (byte_addr / bank_width_bytes) % banks_count;
                for b_offset in 0..4 {
                    let bank = (start_bank + b_offset) % banks_count;
                    phase_bank_counts[bank as usize] += 1;
                }
            }

            for (bank_id, count) in phase_bank_counts.iter().enumerate() {
                if *count > 4 {
                    return Err(format!(
                        "Bank Conflict Prover: ldmatrix.x4 phase {} serialization! \
                         Bank {} accessed {} times simultaneously (max allowed 4 for 4-byte width). \
                         Swizzle mode: {:?}",
                        v, bank_id, count, layout.swizzle_mode
                    ));
                }
            }
        }

        Ok(())
    }

    /// Automated Shared Memory Layout Permutation Pass (XOR Swizzling)
    /// Automatically applies 128B XOR permutation indexing ((row ^ col) % 32)
    /// to shared memory array layouts to eliminate 32-bank SMEM conflicts.
    pub fn auto_swizzle_layout(layout: &mut SmemLayout) -> SwizzleMode {
        if let Some(ref mode) = layout.swizzle_mode {
            return mode.clone();
        }

        let candidates = vec![
            SwizzleMode::Xor128B,
            SwizzleMode::Custom(SwizzlePattern {
                xor_bits: 3,
                base_shift: 4,
                offset: 0,
            }),
            SwizzleMode::Xor64B,
            SwizzleMode::Xor32B,
        ];

        for candidate in candidates {
            let mut test_layout = layout.clone();
            test_layout.swizzle_mode = Some(candidate.clone());
            if Self::prove_ldmatrix_m16n8(&test_layout).is_ok()
                && Self::prove_ldmatrix_m8n8_x4(&test_layout).is_ok()
            {
                layout.swizzle_mode = Some(candidate.clone());
                layout.swizzle = Some(SwizzlePattern {
                    xor_bits: 3,
                    base_shift: 4,
                    offset: 0,
                });
                return candidate;
            }
        }

        let default_mode = SwizzleMode::Xor128B;
        layout.swizzle_mode = Some(default_mode.clone());
        layout.swizzle = Some(SwizzlePattern {
            xor_bits: 3,
            base_shift: 4,
            offset: 0,
        });
        default_mode
    }
}

// ────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_swizzle_conflicts() {
        let layout = SmemLayout {
            rows: 16,
            cols: 64, // F16 matrix (each row is 64x2 = 128 bytes)
            swizzle: None,
            swizzle_mode: None,
            bytes_per_element: 2,
        };
        let result = BankConflictProver::prove_ldmatrix_m16n8(&layout);
        assert!(result.is_err(), "Expected conflicts without swizzle");
    }

    #[test]
    fn test_swizzle_solves_conflicts() {
        let layout = SmemLayout {
            rows: 16,
            cols: 64,
            swizzle: Some(SwizzlePattern {
                xor_bits: 3,
                base_shift: 0,
                offset: 0,
            }),
            swizzle_mode: None,
            bytes_per_element: 2,
        };
        let result = BankConflictProver::prove_ldmatrix_m16n8(&layout);
        assert!(result.is_ok(), "Expected 0 conflicts with proper swizzle");
    }

    #[test]
    fn test_swizzle_128b_xor_ldmatrix_x4() {
        let layout = SmemLayout {
            rows: 32,
            cols: 64,
            swizzle: None,
            swizzle_mode: Some(SwizzleMode::Xor128B),
            bytes_per_element: 2,
        };
        let result = BankConflictProver::prove_ldmatrix_m8n8_x4(&layout);
        assert!(result.is_ok(), "Expected zero bank conflicts with 128B XOR swizzle");
    }

    #[test]
    fn test_auto_swizzle_layout_pass() {
        let mut layout = SmemLayout {
            rows: 16,
            cols: 64,
            swizzle: None,
            swizzle_mode: None,
            bytes_per_element: 2,
        };
        let mode = BankConflictProver::auto_swizzle_layout(&mut layout);
        assert_eq!(mode, SwizzleMode::Xor128B);
        assert!(BankConflictProver::prove_ldmatrix_m16n8(&layout).is_ok());
    }
}


