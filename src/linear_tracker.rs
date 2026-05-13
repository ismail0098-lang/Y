// ============================================================
//  Y-Lang  —  Linear Type Tracker
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
}

/// The LinearTracker manages lexical scopes and enforces
/// linear typing rules on Transfer obligations.
#[derive(Debug, Default)]
pub struct LinearTracker {
    /// A stack of scopes, each containing variable to Obligation mappings.
    scopes: Vec<HashMap<String, Obligation>>,
    pub errors: Vec<String>,
}

impl LinearTracker {
    pub fn new() -> Self {
        Self {
            scopes: vec![HashMap::new()],
            errors: Vec::new(),
        }
    }

    pub fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
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
            scope.insert(
                name.clone(),
                Obligation {
                    name,
                    created_at: span,
                    destination,
                    consumed: false,
                    barrier_synchronized: false,
                },
            );
        }
    }

    /// Mark an obligation as consumed. Returns true if successful, false if it didn't exist or was already consumed.
    pub fn consume_obligation(&mut self, name: &str, use_span: Span) -> bool {
        for scope in self.scopes.iter_mut().rev() {
            if let Some(ob) = scope.get_mut(name) {
                if ob.consumed {
                    self.errors.push(format!(
                        "Line {}: Linear Type Error: `{}` transfer obligation was consumed twice. \
                        It was already awaited previously.",
                        use_span.line, name
                    ));
                    return false;
                } else {
                    ob.consumed = true;
                    ob.barrier_synchronized = false;
                    return true;
                }
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
