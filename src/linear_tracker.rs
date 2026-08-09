// ============================================================
//  Y  —  Linear Type Tracker
//  linear_tracker.rs
//
//  Tracks synchronization obligations in the AST.
//  Whenever an async transfer (e.g. `cp_async`) creates a
//  `Transfer` type, this tracker binds it to the scope.
//  It enforces exactly-once consumption by `pipe.wait`.
// ============================================================

#![allow(dead_code)]

use crate::ast::Span;
use std::collections::HashMap;

/// An obligation to synchronize memory before use.
#[derive(Debug, Clone)]
pub struct Obligation {
    pub name: String,
    pub created_at: Span,
    pub destination: Option<String>,
    pub consumed: bool,
    pub barrier_synchronized: bool,
    /// How many enclosing `if`/`else` branches the *creation* sat inside.
    cond_depth: u32,
    /// How many enclosing loop bodies the *creation* sat inside.
    loop_depth: u32,
}

/// The LinearTracker manages lexical scopes and enforces
/// linear typing rules on Transfer obligations.
///
/// # Control flow
///
/// "Exactly once" is a claim about executions, not about source lines, and the
/// tracker used to check only the latter: it set a `consumed` flag the first
/// time it walked past a `pipe.wait(t)` and never asked whether that statement
/// runs, or how often. Two programs slipped through as a result, both of which
/// compiled clean and emitted a kernel:
///
/// ```text
/// let t = cp_async(A, B, 16);
/// if n { pipe.wait(t); }          // awaited on one path out of two
///
/// let t = cp_async(A, B, 16);
/// for i in 0..4 { pipe.wait(t); } // one copy, four awaits
/// ```
///
/// The first is the failure the tracker exists to prevent - the kernel reads a
/// destination whose `cp.async` may still be in flight. The second is the
/// double-consume it already rejects when written twice in a row, hidden behind
/// a loop header. Neither is caught by counting flags, so the tracker now
/// records the conditional and loop nesting depth at creation and compares it
/// against the depth at consumption. A consume deeper than its creation is
/// refused, with the two cases reported separately because the fixes differ:
/// hoist the wait out of the branch, versus move the copy into the loop.
#[derive(Debug, Default)]
pub struct LinearTracker {
    /// A stack of scopes, each containing variable to Obligation mappings.
    scopes: Vec<HashMap<String, Obligation>>,
    /// Enclosing `if`/`else` branch nesting at the current walk position.
    cond_depth: u32,
    /// Enclosing loop-body nesting at the current walk position.
    loop_depth: u32,
    pub errors: Vec<String>,
}

impl LinearTracker {
    pub fn new() -> Self {
        Self {
            scopes: vec![HashMap::new()],
            cond_depth: 0,
            loop_depth: 0,
            errors: Vec::new(),
        }
    }

    pub fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    /// Called by the type checker around an `if`/`else` branch body.
    pub fn enter_conditional(&mut self) {
        self.cond_depth += 1;
    }

    pub fn exit_conditional(&mut self) {
        self.cond_depth = self.cond_depth.saturating_sub(1);
    }

    /// Called by the type checker around a loop body.
    pub fn enter_loop(&mut self) {
        self.loop_depth += 1;
    }

    pub fn exit_loop(&mut self) {
        self.loop_depth = self.loop_depth.saturating_sub(1);
    }

    /// Pops the top scope. If any obligation was left unconsumed, returns an error.
    pub fn pop_scope(&mut self) {
        if let Some(scope) = self.scopes.pop() {
            for (name, ob) in scope {
                if !ob.consumed {
                    self.errors.push(format!(
                        "Line {}: Linear Type Error: `{}` is a Transfer obligation that was never consumed. \
                         You must call `pipe.wait({})` before it goes out of scope.",
                        ob.created_at.line, name, name
                    ));
                }
            }
        }
    }

    /// Register a new linear obligation in the current scope.
    pub fn register_obligation(&mut self, name: String, span: Span, destination: Option<String>) {
        if let Some(scope) = self.scopes.last_mut() {
            // Shadowing an existing obligation unconsumed is also an error.
            if let Some(prev) = scope.get(&name) {
                if !prev.consumed {
                    self.errors.push(format!(
                        "Line {}: Linear Type Error: `{}` was reassigned before its previous obligation was consumed.",
                        span.line, name
                    ));
                }
            }
            let (cond_depth, loop_depth) = (self.cond_depth, self.loop_depth);
            scope.insert(
                name.clone(),
                Obligation {
                    name,
                    created_at: span,
                    destination,
                    consumed: false,
                    barrier_synchronized: false,
                    cond_depth,
                    loop_depth,
                },
            );
        }
    }

    /// Mark an obligation as consumed. Returns true if successful, false if it didn't exist or was already consumed.
    pub fn consume_obligation(&mut self, name: &str, use_span: Span) -> bool {
        let (here_cond, here_loop) = (self.cond_depth, self.loop_depth);
        for scope in self.scopes.iter_mut().rev() {
            if let Some(ob) = scope.get_mut(name) {
                if ob.consumed {
                    self.errors.push(format!(
                        "Line {}: Linear Type Error: `{}` transfer obligation was consumed twice. \
                        It was already awaited previously.",
                        use_span.line, name
                    ));
                    return false;
                }

                // A wait nested inside a loop the copy is not inside runs once
                // per iteration against a single `cp.async`, which is the
                // double-consume above with a loop header in front of it.
                if here_loop > ob.loop_depth {
                    let msg = format!(
                        "Line {}: Linear Type Error: `{}` is awaited inside a loop but was \
                         created outside it, so one transfer would be awaited once per \
                         iteration. Move the `cp_async(...)` into the loop body, or hoist \
                         the `pipe.wait({})` out of it.",
                        use_span.line, name, name
                    );
                    self.errors.push(msg);
                    // Still mark it consumed: leaving it pending would add a
                    // second, misleading "never consumed" error at scope exit
                    // for the same underlying mistake.
                    ob.consumed = true;
                    ob.barrier_synchronized = false;
                    return false;
                }

                // A wait nested inside a branch the copy is not inside is
                // skipped whenever that branch is not taken, so the transfer is
                // read while still in flight. "Exactly once" has to mean on
                // every path, not on some path.
                if here_cond > ob.cond_depth {
                    let msg = format!(
                        "Line {}: Linear Type Error: `{}` is awaited inside a conditional \
                         branch, so on the paths that skip the branch the transfer is never \
                         awaited and its destination is read while still in flight. Move the \
                         `pipe.wait({})` outside the `if`.",
                        use_span.line, name, name
                    );
                    self.errors.push(msg);
                    ob.consumed = true;
                    ob.barrier_synchronized = false;
                    return false;
                }

                ob.consumed = true;
                ob.barrier_synchronized = false;
                return true;
            }
        }

        // Not a tracked linear transfer (or it was never defined/is just a normal variable)
        // Handled by standard type checking elsewhere.
        true
    }

    pub fn is_tracked_obligation(&self, name: &str) -> bool {
        self.scopes
            .iter()
            .rev()
            .any(|scope| scope.contains_key(name))
    }

    pub fn synchronize_barrier(&mut self) {
        for scope in &mut self.scopes {
            for obligation in scope.values_mut() {
                if obligation.consumed {
                    obligation.barrier_synchronized = true;
                }
            }
        }
    }

    pub fn require_destination_ready(&mut self, destination: &str, use_span: Span) -> bool {
        let mut pending_wait = Vec::new();
        let mut pending_barrier = Vec::new();

        for scope in self.scopes.iter().rev() {
            for obligation in scope.values() {
                if obligation.destination.as_deref() != Some(destination) {
                    continue;
                }

                if !obligation.consumed {
                    pending_wait.push(obligation.name.clone());
                } else if !obligation.barrier_synchronized {
                    pending_barrier.push(obligation.name.clone());
                }
            }
        }

        if pending_wait.is_empty() && pending_barrier.is_empty() {
            return true;
        }

        pending_wait.sort();
        pending_wait.dedup();
        pending_barrier.sort();
        pending_barrier.dedup();

        let message = if !pending_wait.is_empty() && !pending_barrier.is_empty() {
            format!(
                "Line {}: Linear Type Error: shared destination `{}` cannot be read because Transfer obligation(s) [{}] are still pending and awaited obligation(s) [{}] have not passed `barrier::sync()`. Call `pipe.wait(...)` and then `barrier::sync()` first.",
                use_span.line,
                destination,
                pending_wait.join(", "),
                pending_barrier.join(", ")
            )
        } else if !pending_wait.is_empty() {
            format!(
                "Line {}: Linear Type Error: shared destination `{}` cannot be read while Transfer obligation(s) [{}] are still pending. Call `pipe.wait(...)` and then `barrier::sync()` before reading it.",
                use_span.line,
                destination,
                pending_wait.join(", ")
            )
        } else {
            format!(
                "Line {}: Linear Type Error: shared destination `{}` cannot be read because awaited Transfer obligation(s) [{}] have not passed `barrier::sync()`. Synchronize the pipeline before reading shared memory.",
                use_span.line,
                destination,
                pending_barrier.join(", ")
            )
        };

        self.errors.push(message);
        false
    }

    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }
}

