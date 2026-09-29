(** * The ZK backend's control-flow lowering, mechanically verified.

    Y's R1CS backend has no branches: a circuit is a fixed system of
    constraints, so [if] must be lowered to arithmetic and [return] to a value.
    Getting that wrong does not crash and does not fail to compile - it emits a
    circuit computing a DIFFERENT function, which Groth16 then proves perfectly
    happily. Three such bugs were found in this one lowering: two with Z3, and
    the third by writing down the semantics for this file.

    What is proved: for a language of [return], [if] and sequencing, the
    PREDICATED lowering agrees with the operational semantics on every program,
    every environment and every input. [low_correct] says the circuit's output
    is bound to exactly the value the program returns, and that the "did it
    return" flag is 0 exactly when control falls off the end.

    The two earlier lowerings are formalised too and refuted by machine-checked
    counterexamples: [low_last] (what the emitter did) and [low_tail] (what it
    did after the first fix).

    *** What this does NOT prove. ***

    This is a proof about a MODEL of the lowering, not about [zk_emitter.rs].
    There is no extraction and no refinement proof, so a divergence between
    this model and the Rust is not detected here; [tests/zk_control_flow.rs]
    is what ties the two together, and it exercises the same cases the lemmas
    below name. Two further abstractions are deliberate:

    - The field is modelled as a commutative ring, taken to be [Z]. Every step
      below is a ring identity - no inverse, no division - so the results
      transfer to BN254's Fr. What is LOST is any claim about range or
      reduction: [Z] is infinite and a field is not.
    - Constraint emission is modelled as evaluation. [low] computes the value
      the output wire is constrained to equal, rather than emitting
      constraints. That is the right abstraction for "is the output bound to
      the right value" and says nothing about constraint count, nor about
      whether an adversarial prover could satisfy the system some other way.

    Build:  coqc proofs/ZkControlFlow.v      (Rocq 9.1)
*)

From Stdlib Require Import ZArith.
Open Scope Z_scope.

(** ** The language

    Sequencing is a constructor rather than a list. A block [s1; s2; s3] is
    [SSeq s1 (SSeq s2 (SSeq s3 SNop))]. This is the same language, and it
    keeps [stmt] a plain inductive, so the generated induction principle is
    strong enough - with [list stmt] nested inside [stmt] it is not, and every
    proof needs a hand-rolled mutual scheme. *)

Definition env := nat -> Z.

Inductive expr : Type :=
| EConst : Z -> expr
| EVar   : nat -> expr
| EAdd   : expr -> expr -> expr
| EMul   : expr -> expr -> expr.

Fixpoint eval (e : env) (x : expr) : Z :=
  match x with
  | EConst z => z
  | EVar n => e n
  | EAdd a b => eval e a + eval e b
  | EMul a b => eval e a * eval e b
  end.

Inductive stmt : Type :=
| SNop : stmt                            (** the empty block *)
| SRet : expr -> stmt
| SIf  : expr -> stmt -> stmt -> stmt
| SSeq : stmt -> stmt -> stmt.

(** ** Operational semantics

    [run e s = Some v] means [s] returns [v]; [None] means control falls off
    the end. A [return] TERMINATES - what follows it is unreachable - and a
    branch that falls through continues with whatever comes after the [if]. *)

Fixpoint run (e : env) (s : stmt) : option Z :=
  match s with
  | SNop => None
  | SRet x => Some (eval e x)
  | SIf c t f => if Z.eqb (eval e c) 1 then run e t else run e f
  | SSeq a b => match run e a with
                | Some v => Some v
                | None => run e b
                end
  end.

(** ** The arithmetic selector

    An R1CS multiplexer is the single constraint [c * (a - b) = out - b]. It
    selects a branch only when [c] is 0 or 1; for anything else it
    interpolates. *)

Definition mux (c a b : Z) : Z := c * (a - b) + b.

Lemma mux_1 : forall a b, mux 1 a b = a.
Proof. intros; unfold mux; ring. Qed.

Lemma mux_0 : forall a b, mux 0 a b = b.
Proof. intros; unfold mux; ring. Qed.

(** The condition really must be a bit. This is the bug that made
    [if a { return 5; } return 9;] emit [9 - 4a], answering 1 for [a = 2] -
    a value neither branch can return. It is now a booleanity constraint on
    the condition, which makes such a circuit unsatisfiable rather than wrong. *)
Example mux_needs_a_boolean : mux 2 5 9 = 1.
Proof. reflexivity. Qed.

(** Every [if] condition in the program evaluates to 0 or 1. *)
Fixpoint bools (e : env) (s : stmt) : Prop :=
  match s with
  | SNop => True
  | SRet _ => True
  | SIf c t f => (eval e c = 0 \/ eval e c = 1) /\ bools e t /\ bools e f
  | SSeq a b => bools e a /\ bools e b
  end.

(** ** Lowering 1: what the emitter did

    [emit_block] ran every statement and kept the LAST return it saw, so a
    [return] neither terminated the block nor took precedence over a later
    one. In this encoding: the right-hand side of a [SSeq] wins. *)

Fixpoint low_last (e : env) (s : stmt) : option Z :=
  match s with
  | SNop => None
  | SRet x => Some (eval e x)
  | SIf c t f =>
      match low_last e t, low_last e f with
      | Some vt, Some vf => Some (mux (eval e c) vt vf)
      | Some vt, None => Some (mux (eval e c) vt 0)
      | None, Some vf => Some (mux (eval e c) 0 vf)
      | None, None => None
      end
  | SSeq a b => match low_last e b with
                | Some v => Some v
                | None => low_last e a
                end
  end.

(** ** Lowering 2: after the first fix

    Stop at the first return, and multiplex a one-sided return against the
    value of the REST of the block. Correct for a flat block, and still wrong
    once the [if] is nested: an inner [if] that falls through reports the zero
    its own empty tail supplies, instead of telling the enclosing block that
    control fell through. *)

Fixpoint low_tail (e : env) (s : stmt) : option Z :=
  match s with
  | SNop => None
  | SRet x => Some (eval e x)
  | SIf c t f =>
      match low_tail e t, low_tail e f with
      | Some vt, Some vf => Some (mux (eval e c) vt vf)
      | Some vt, None => Some (mux (eval e c) vt 0)
      | None, Some vf => Some (mux (eval e c) 0 vf)
      | None, None => None
      end
  | SSeq (SIf c t f) rest =>
      let tl := match low_tail e rest with Some v => v | None => 0 end in
      match low_tail e t, low_tail e f with
      | Some vt, Some vf => Some (mux (eval e c) vt vf)
      | Some vt, None => Some (mux (eval e c) vt tl)
      | None, Some vf => Some (mux (eval e c) tl vf)
      | None, None => low_tail e rest
      end
  | SSeq a rest => match low_tail e a with
                   | Some v => Some v
                   | None => low_tail e rest
                   end
  end.

(** ** Lowering 3: predicated returns

    Each statement yields a PAIR: a flag saying whether it returned, and the
    value it returned. "Did it return" is exactly the information the two
    lowerings above threw away, and it is what lets an inner [if] report
    fall-through to its enclosing block.

    In R1CS this costs one extra multiplexer and one extra product per
    sequencing point that can conditionally return - and nothing at all for a
    block whose statements return unconditionally, where the flag folds to a
    constant. *)

Fixpoint low (e : env) (s : stmt) : Z * Z :=
  match s with
  | SNop => (0, 0)
  | SRet x => (1, eval e x)
  | SIf c t f =>
      (mux (eval e c) (fst (low e t)) (fst (low e f)),
       mux (eval e c) (snd (low e t)) (snd (low e f)))
  | SSeq a b =>
      let fa := fst (low e a) in
      let va := snd (low e a) in
      (fa + (1 - fa) * fst (low e b), mux fa va (snd (low e b)))
  end.

(** ** Correctness

    One statement covering both halves: when the program returns, the flag is
    1 and the value is exactly what it returns; when control falls off the end,
    the flag is 0. *)

Definition agrees (e : env) (s : stmt) : Prop :=
  match run e s with
  | Some v => low e s = (1, v)
  | None => fst (low e s) = 0
  end.

Theorem low_correct : forall s e, bools e s -> agrees e s.
Proof.
  induction s as [ | x | c t IHt f IHf | a IHa b IHb ]; intros e Hb.
  - (* SNop *) unfold agrees. simpl. reflexivity.
  - (* SRet *) unfold agrees. simpl. reflexivity.
  - (* SIf *)
    simpl in Hb. destruct Hb as [Hc [Ht Hf]].
    specialize (IHt e Ht). specialize (IHf e Hf).
    unfold agrees in *. simpl.
    destruct (low e t) as [ft vt] eqn:Lt.
    destruct (low e f) as [ff vf] eqn:Lf.
    simpl in *.
    destruct Hc as [Hc | Hc]; rewrite Hc; simpl.
    + (* condition 0: the else branch runs *)
      rewrite !mux_0.
      destruct (run e f) as [v | ]; [ injection IHf as A B; subst | subst ];
        reflexivity.
    + (* condition 1: the then branch runs *)
      rewrite !mux_1.
      destruct (run e t) as [v | ]; [ injection IHt as A B; subst | subst ];
        reflexivity.
  - (* SSeq *)
    simpl in Hb. destruct Hb as [Ha Hbb].
    specialize (IHa e Ha). specialize (IHb e Hbb).
    unfold agrees in *. simpl.
    destruct (low e a) as [fa va] eqn:La.
    destruct (low e b) as [fb vb] eqn:Lb.
    simpl in *.
    destruct (run e a) as [v | ].
    + (* a returns: b is unreachable, and the flag folds to the constant 1 *)
      injection IHa as A B; subst.
      apply injective_projections; cbn; try unfold mux; ring.
    + (* a falls through: the answer is b's *)
      subst fa.
      destruct (run e b) as [v | ].
      * injection IHb as A B; subst.
        apply injective_projections; cbn; try unfold mux; ring.
      * cbn. subst fb. ring.
Qed.

(** A corollary in the form the emitter cares about: the output wire is bound
    to the value the program returns. *)
Corollary output_is_the_returned_value :
  forall s e v, bools e s -> run e s = Some v -> snd (low e s) = v.
Proof.
  intros s e v Hb Hr.
  pose proof (low_correct s e Hb) as H. unfold agrees in H.
  rewrite Hr in H. rewrite H. reflexivity.
Qed.

(** ** The three bugs, as machine-checked counterexamples

    Each is a program on which the older lowering disagrees with the
    semantics. All are closed terms, so [reflexivity] decides them. *)

(** Environment: variable 0 is true, variable 1 is false. *)
Definition e01 : env := fun n => match n with 0%nat => 1 | _ => 0 end.

(** *** Bug 1a: [return 1; return 2;] *)
Definition p_dead : stmt := SSeq (SRet (EConst 1)) (SRet (EConst 2)).

Example dead_code_semantics : run e01 p_dead = Some 1.
Proof. reflexivity. Qed.

Example low_last_takes_the_unreachable_return : low_last e01 p_dead = Some 2.
Proof. reflexivity. Qed.

Example low_gets_dead_code_right : low e01 p_dead = (1, 1).
Proof. reflexivity. Qed.

(** *** Bug 1b: [if c { return 1; } return 0;] with [c] true.

    The emitter emitted a circuit for the constant 0; Z3 confirmed on the
    real artifact that no satisfying assignment had a non-zero output. *)
Definition p_flat : stmt :=
  SSeq (SIf (EVar 0) (SRet (EConst 1)) SNop) (SRet (EConst 0)).

Example flat_semantics : run e01 p_flat = Some 1.
Proof. reflexivity. Qed.

Example low_last_collapses_to_the_tail : low_last e01 p_flat = Some 0.
Proof. reflexivity. Qed.

Example low_gets_the_flat_case_right : low e01 p_flat = (1, 1).
Proof. reflexivity. Qed.

(** *** Bug 2: the same shape NESTED, which survived the first fix.

    [if c { if d { return 1; } } return 7;] with [c] true and [d] false. The
    program returns 7; [low_tail] answers 0, because the inner [if] muxed its
    fall-through against its own empty tail instead of propagating it. This is
    the case the proof attempt above found, and it reproduced on the real
    compiler before this file was finished. *)
Definition p_nested : stmt :=
  SSeq (SIf (EVar 0) (SIf (EVar 1) (SRet (EConst 1)) SNop) SNop)
       (SRet (EConst 7)).

Example nested_semantics : run e01 p_nested = Some 7.
Proof. reflexivity. Qed.

Example low_tail_is_wrong_when_nested : low_tail e01 p_nested = Some 0.
Proof. reflexivity. Qed.

Example low_gets_the_nested_case_right : low e01 p_nested = (1, 7).
Proof. reflexivity. Qed.

(** The flat case is exactly where [low_tail] looks correct, which is why it
    survived: the obvious test passes. *)
Example low_tail_is_right_on_the_flat_case : low_tail e01 p_flat = Some 1.
Proof. reflexivity. Qed.

(** ** Unrolled loops are covered by the same theorem

    The emitter fully unrolls [for], so a loop body is a statement SEQUENCE and
    nothing about it is new: [low_correct] applies as it stands. This matters
    because the loop arm was a THIRD site with the same bug - it discarded its
    body's result entirely, so [for i in 0..4 { if c { return 1; } } return 9;]
    emitted the constant 9.

    Two unrolled iterations of that program, with the condition true and then
    false: *)

Definition p_loop (c : expr) : stmt :=
  SSeq (SIf c (SRet (EConst 1)) SNop)
       (SSeq (SIf c (SRet (EConst 1)) SNop)
             (SRet (EConst 9))).

Definition e_true  : env := fun _ => 1.
Definition e_false : env := fun _ => 0.

Example loop_returns_on_the_first_iteration :
  run e_true (p_loop (EVar 0)) = Some 1 /\ low e_true (p_loop (EVar 0)) = (1, 1).
Proof. split; reflexivity. Qed.

Example loop_falls_through_to_the_tail :
  run e_false (p_loop (EVar 0)) = Some 9 /\ low e_false (p_loop (EVar 0)) = (1, 9).
Proof. split; reflexivity. Qed.

(** Note the loop bug was NOT a wrong formula - [low_tail] gets this shape
    right, because an unrolled loop is a flat sequence and flat sequences are
    exactly where it works. The emitter's loop arm simply never consumed the
    body's result at all, which no choice of lowering can repair. Worth
    recording so the fix is not mis-attributed: two of the three sites needed a
    better lowering, and the third needed the lowering to be CALLED. *)
Example low_tail_is_right_on_this_shape :
  low_tail e_false (p_loop (EVar 0)) = Some 9.
Proof. reflexivity. Qed.

(** ** Trust base

    Nothing here is admitted, assumed, or axiomatised: the proofs rest only on
    Rocq's kernel and the [ring] decision procedure for [Z]. *)
Print Assumptions low_correct.
Print Assumptions output_is_the_returned_value.
