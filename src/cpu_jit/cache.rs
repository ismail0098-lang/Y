//! Bounded process-local reuse. Callers retain executable code after eviction.
use super::{CpuJit, JitError, JitOptions};
use sha2::{Digest, Sha256};
use std::collections::VecDeque;
use std::rc::Rc;

pub struct CpuJitCache {
    capacity: usize,
    entries: VecDeque<([u8; 32], Rc<CpuJit>)>,
    hits: u64,
    misses: u64,
}

impl CpuJitCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entries: VecDeque::new(),
            hits: 0,
            misses: 0,
        }
    }

    /// Reuse the same source/optimization pair; otherwise compile once.
    /// Native globals and runtime state are shared when a JIT is reused.
    pub fn compile(&mut self, source: &str, options: JitOptions) -> Result<Rc<CpuJit>, JitError> {
        let mut hash = Sha256::new();
        hash.update(b"Y CPU JIT process cache v13\0");
        hash.update([
            options.opt_level,
            u8::from(options.training_opt_level.is_some()),
            options.training_opt_level.unwrap_or(0),
            u8::from(options.profile_edge_counters),
            u8::from(options.profile_loop_edge_counters),
            u8::from(options.final_loop_unrolling),
            u8::from(options.final_unroll_outer_loops),
            u8::from(options.codegen_opt_level.is_some()),
            options.codegen_opt_level.unwrap_or(0),
            u8::from(options.verify_each_pass),
            u8::from(options.recognize_rotates),
            u8::from(options.optimize_runtime),
            u8::from(options.optimize_runtime_mutations),
            u8::from(options.optimize_runtime_copies),
            u8::from(options.optimize_helper_effects),
            u8::from(options.optimize_call_adapters),
            u8::from(options.profile_loop_controls),
        ]);
        hash.update(source.as_bytes());
        let key: [u8; 32] = hash.finalize().into();
        if let Some(index) = self
            .entries
            .iter()
            .position(|(candidate, _)| *candidate == key)
        {
            let entry = self.entries.remove(index).unwrap();
            let jit = Rc::clone(&entry.1);
            self.entries.push_back(entry);
            self.hits += 1;
            return Ok(jit);
        }
        self.misses += 1;
        let jit = Rc::new(CpuJit::compile_with_options(source, options)?);
        if self.capacity != 0 {
            if self.entries.len() == self.capacity {
                self.entries.pop_front();
            }
            self.entries.push_back((key, Rc::clone(&jit)));
        }
        Ok(jit)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    pub fn clear(&mut self) {
        self.entries.clear();
    }
    pub fn hits(&self) -> u64 {
        self.hits
    }
    pub fn misses(&self) -> u64 {
        self.misses
    }
}

impl Default for CpuJitCache {
    fn default() -> Self {
        Self::new(8)
    }
}
