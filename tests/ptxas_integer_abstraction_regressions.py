"""Solver controls for integer abstraction congruence and exact encoding.

These checks establish arithmetic-model properties without GPU execution.
An abstraction's SAT result only shows a missing equality; exact-encoding
controls must agree with the concrete bitvector solver on SAT and UNSAT.
"""

from pathlib import Path
import sys
import unittest

from verification_unittest import main as verification_main


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools' / 'ptxas_tval'))

try:
    import z3
except ImportError:
    HAS_Z3 = False
else:
    HAS_Z3 = True
    import intenc
    import mulmode


@unittest.skipUnless(HAS_Z3, 'integer abstraction checks require z3-solver')
class MultiplierCongruence(unittest.TestCase):
    def setUp(self):
        self.x, self.y, self.u, self.v = z3.BitVecs(
            'imul_alias_x imul_alias_y imul_alias_u imul_alias_v', 32)

    def check(self, left, right, expected, *assumptions):
        solver = z3.Solver()
        solver.set(timeout=5000)
        solver.add(*assumptions, left != right)
        self.assertEqual(solver.check(), expected, solver.reason_unknown())

    def test_aliases_and_symbolic_substitution_preserve_both_halves(self):
        for mode in ('uf', 'wide'):
            multiply = mulmode.MODES[mode]()
            for kind in ('lo', 'hi'):
                with self.subTest(mode=mode, kind=kind):
                    original = multiply(kind, self.x, self.y)
                    self.check(original, multiply(kind, self.x + 0, self.y), z3.unsat)
                    substituted = z3.substitute(
                        multiply(kind, self.u, self.v),
                        (self.u, self.y), (self.v, self.x))
                    self.check(original, substituted, z3.unsat)
                    self.check(original, multiply(kind, self.v, self.u), z3.unsat,
                               self.x == self.u, self.y == self.v)

    def test_other_products_remain_distinguishable(self):
        for mode in ('uf', 'wide'):
            multiply = mulmode.MODES[mode]()
            for kind in ('lo', 'hi'):
                with self.subTest(mode=mode, kind=kind):
                    self.check(multiply(kind, self.x, self.y),
                               multiply(kind, self.u, self.y), z3.sat)

    def test_wide_halves_share_one_value_after_aliases(self):
        multiply = mulmode.wide_factory()
        original = z3.Concat(multiply('hi', self.x, self.y),
                             multiply('lo', self.x, self.y))
        aliases = z3.Concat(multiply('hi', self.y, self.x + 0),
                            multiply('lo', self.y + 0, self.x))
        self.check(original, aliases, z3.unsat)
        # Swapping high and low halves must remain detectable.
        swapped = z3.Concat(multiply('lo', self.x, self.y),
                            multiply('hi', self.x, self.y))
        self.check(original, swapped, z3.sat)

    def test_literal_and_power_of_two_paths_match_unsigned_multiplication(self):
        edges = (0, 1, 2, 0x7fffffff, 0x80000000, 0xffffffff)
        for mode in ('uf', 'wide'):
            multiply = mulmode.MODES[mode]()
            for kind in ('lo', 'hi'):
                for a in edges:
                    for b in edges:
                        with self.subTest(mode=mode, kind=kind, a=a, b=b):
                            av, bv = z3.BitVecVal(a, 32), z3.BitVecVal(b, 32)
                            self.assertTrue(z3.is_true(z3.simplify(
                                multiply(kind, av, bv) == mulmode.direct(kind, av, bv))))
                for shift in (0, 1, 15, 31):
                    with self.subTest(mode=mode, kind=kind, shift=shift):
                        literal = z3.BitVecVal(1 << shift, 32)
                        self.check(multiply(kind, self.x, literal),
                                   mulmode.direct(kind, self.x, literal), z3.unsat)


@unittest.skipUnless(HAS_Z3, 'exact integer encoding checks require z3-solver')
class ExactIntegerEncoding(unittest.TestCase):
    def agrees(self, formulas, expected):
        solver = z3.Solver()
        solver.set(timeout=5000)
        solver.add(*formulas)
        self.assertEqual(str(solver.check()), expected, solver.reason_unknown())
        for rewrite in (True, False):
            with self.subTest(rewrite=rewrite):
                self.assertEqual(intenc.check(formulas, 5, rewrite=rewrite), expected)

    def concrete(self, term, assignments, expected):
        assumptions = [symbol == value for symbol, value in assignments]
        literal = z3.BitVecVal(expected, term.size())
        self.agrees(assumptions + [term == literal], 'sat')
        self.agrees(assumptions + [term != literal], 'unsat')

    def test_same_named_input_widths_are_independent(self):
        byte = z3.BitVec('same_input', 8)
        word = z3.BitVec('same_input', 16)
        self.agrees([byte == 1, word == 2], 'sat')
        self.agrees([byte == 1, word == 256], 'sat')
        self.agrees([byte == 1, byte == 2, word == 256], 'unsat')

    def test_same_named_boolean_and_bitvector_inputs_are_independent(self):
        bits = z3.BitVec('same_boolean_input', 8)
        flag = z3.Bool('same_boolean_input')
        self.agrees([bits == 1, z3.Not(flag)], 'sat')
        self.agrees([bits == 0, flag], 'sat')
        self.agrees([bits == 1, flag, z3.Not(flag)], 'unsat')

    def test_identically_printed_integer_and_string_symbols_are_independent(self):
        numeric = z3.BitVec(1, 8)
        textual = z3.BitVec('k!1', 8)
        numeric_bool, textual_bool = z3.Bool(1), z3.Bool('k!1')
        self.assertEqual(str(numeric), str(textual))
        self.assertFalse(numeric.eq(textual))
        self.agrees([numeric == 1, textual == 2, numeric_bool,
                     z3.Not(textual_bool)], 'sat')
        self.agrees([numeric == 1, numeric == 2, textual == 2], 'unsat')

    def test_function_overloads_preserve_input_and_output_sorts(self):
        for change_domain in (False, True):
            with self.subTest(change_domain=change_domain):
                domain = 16 if change_domain else 8
                result = 8 if change_domain else 16
                f = z3.Function('overloaded_product', z3.BitVecSort(8), z3.BitVecSort(8))
                g = z3.Function('overloaded_product', z3.BitVecSort(domain),
                                z3.BitVecSort(result))
                x, y = z3.BitVec('argument8', 8), z3.BitVec('argument_other', domain)
                self.agrees([x == 3, y == 3, f(x) == 1, g(y) == 2], 'sat')
                self.agrees([x == 3, f(x) == 1, f(x) == 2], 'unsat')

    def test_array_overloads_preserve_index_and_element_sorts(self):
        for change_domain in (False, True):
            with self.subTest(change_domain=change_domain):
                domain = 16 if change_domain else 8
                result = 8 if change_domain else 16
                a = z3.Array('overloaded_memory', z3.BitVecSort(8), z3.BitVecSort(8))
                b = z3.Array('overloaded_memory', z3.BitVecSort(domain), z3.BitVecSort(result))
                x, y = z3.BitVec('index8', 8), z3.BitVec('index_other', domain)
                self.agrees([x == 3, y == 3, z3.Select(a, x) == 1,
                             z3.Select(b, y) == 2], 'sat')
                self.agrees([x == 3, z3.Select(a, x) == 1,
                             z3.Select(a, x) == 2], 'unsat')

    def test_array_reads_do_not_alias_same_named_source_functions(self):
        index = z3.BitVec('heap_index', 8)
        memory = z3.Array('heap', z3.BitVecSort(8), z3.BitVecSort(8))
        ordinary = z3.Function('sel_heap', z3.BitVecSort(8), z3.BitVecSort(8))
        self.agrees([z3.Select(memory, index) == 1, ordinary(index) == 2], 'sat')

    def test_source_names_do_not_capture_encoder_temporaries(self):
        # Force an actual wrap with rewrite=False while source inputs use the
        # exact spellings of generated quotients, residues and other symbols.
        x = z3.BitVec('plain', 8)
        names = ('_ie1_k', '_ie2_r', '_iv_plain_8', '_if_function_0', '_ib_flag')
        source = [z3.BitVec(name, 8) for name in names]
        function = z3.Function('function', z3.BitVecSort(8), z3.BitVecSort(8))
        flag = z3.Bool('_ie2_r')
        formulas = [x == 255, x + 1 == 0, function(x) == 23, flag]
        formulas += [value == 7 + i for i, value in enumerate(source)]
        self.agrees(formulas, 'sat')

    def test_models_retain_all_overloaded_input_values(self):
        byte = z3.BitVec('model_input', 8)
        word = z3.BitVec('model_input', 16)
        single = z3.BitVec('ordinary_model_input', 32)
        result, model = intenc.check([byte == 1, word == 256, single == 7], 5,
                                    want_model=True)
        self.assertEqual(result, 'sat')
        self.assertEqual(model, {'model_input:BV8': 1, 'model_input:BV16': 256,
                                 'ordinary_model_input': 7})

    def test_model_labels_cannot_capture_source_names(self):
        numeric, textual = z3.BitVec(1, 8), z3.BitVec('k!1', 8)
        source_label = z3.BitVec('k!1:BV8', 8)
        result, model = intenc.check([numeric == 1, textual == 2, source_label == 3], 5,
                                    want_model=True)
        self.assertEqual(result, 'sat')
        self.assertEqual(model, {'k!1:BV8#1': 1, 'k!1:BV8#2': 2, 'k!1:BV8': 3})

    def test_quantifiers_and_bound_variables_refuse(self):
        x = z3.BitVec('quantified_input', 4)
        quantified = ((z3.ForAll([x], x == 0), z3.unsat),
                      (z3.Exists([x], z3.ULT(x, 3)), z3.sat))
        for formula, expected in quantified:
            with self.subTest(formula=formula):
                solver = z3.Solver()
                solver.set(timeout=5000)
                solver.add(formula)
                self.assertEqual(solver.check(), expected, solver.reason_unknown())
                for rewrite in (True, False):
                    self.assertEqual(intenc.check([formula], 5, rewrite=rewrite), 'unknown')
                with self.assertRaisesRegex(intenc.Unsupported, 'non-application boolean'):
                    intenc.Encoder(formula.ctx).bool(formula)
        bound_bits = z3.Var(0, z3.BitVecSort(4))
        bound_bool = z3.Var(0, z3.BoolSort())
        for rewrite in (True, False):
            self.assertEqual(intenc.check([bound_bits == 0], 5, rewrite=rewrite), 'unknown')
            self.assertEqual(intenc.check([bound_bool], 5, rewrite=rewrite), 'unknown')

    def test_nonboolean_roots_and_unsupported_equalities_refuse(self):
        unsupported = (z3.BitVec('root_bits', 8), z3.Int('root_int'),
                       z3.Real('root_real'), z3.FP('root_float', z3.Float32()),
                       z3.Array('root_array', z3.BitVecSort(8), z3.BitVecSort(8)))
        for term in unsupported:
            with self.subTest(sort=term.sort()):
                for rewrite in (True, False):
                    self.assertEqual(intenc.check([term], 5, rewrite=rewrite), 'unknown')
                with self.assertRaisesRegex(intenc.Unsupported, 'non-boolean term'):
                    intenc.Encoder(term.ctx).bool(term)
        for term in unsupported[1:]:
            other = z3.Const('other_' + str(term), term.sort())
            self.assertEqual(intenc.check([term == other], 5, rewrite=False), 'unknown')

    def test_nonbare_array_reads_refuse_before_rewriting(self):
        index = z3.BitVec('array_form_index', 4)
        memory = z3.Array('array_form_memory', index.sort(), index.sort())
        forms = ((z3.Lambda([index], index + 1), 2),
                 (z3.K(index.sort(), z3.BitVecVal(2, 4)), 2),
                 (z3.Store(memory, index, 2), 2))
        for array, expected in forms:
            with self.subTest(array=array):
                read = z3.Select(array, index)
                with self.assertRaisesRegex(intenc.Unsupported, 'non-bare array'):
                    intenc.Encoder(read.ctx).bv(read)
                for equal, result in ((True, 'sat'), (False, 'unsat')):
                    relation = read == expected if equal else read != expected
                    formulas = [index == 1, relation]
                    solver = z3.Solver()
                    solver.set(timeout=5000)
                    solver.add(*formulas)
                    self.assertEqual(str(solver.check()), result, solver.reason_unknown())
                    self.assertEqual(intenc.check(formulas, 5, rewrite=False), 'unknown')
                    # Rewriting may eliminate the unsupported array entirely;
                    # the resulting scalar formula keeps its exact meaning.
                    self.assertEqual(intenc.check(formulas, 5), result)

    def test_boolean_uf_and_boolean_array_results_refuse(self):
        x = z3.BitVec('boolean_result_index', 4)
        predicate = z3.Function('boolean_result_function', x.sort(), z3.BoolSort())
        memory = z3.Array('boolean_result_array', x.sort(), z3.BoolSort())
        for term in (predicate(x), z3.Select(memory, x)):
            with self.subTest(term=term):
                solver = z3.Solver()
                solver.set(timeout=5000)
                solver.add(x == 1, term)
                self.assertEqual(solver.check(), z3.sat, solver.reason_unknown())
                for rewrite in (True, False):
                    self.assertEqual(intenc.check([x == 1, term], 5, rewrite=rewrite), 'unknown')
                with self.assertRaisesRegex(intenc.Unsupported, 'boolean operator'):
                    intenc.Encoder(term.ctx).bool(term)

    def test_supported_boolean_function_arguments_preserve_congruence(self):
        flag = z3.Bool('function_boolean_argument')
        function = z3.Function('boolean_argument_function', z3.BoolSort(), z3.BitVecSort(8))
        self.agrees([flag, function(flag) == 1, function(z3.BoolVal(False)) == 2], 'sat')
        self.agrees([flag, function(flag) != function(z3.BoolVal(True))], 'unsat')

    def test_nary_arithmetic_matches_modular_oracle(self):
        for width in (1, 4, 8):
            mask, high = (1 << width) - 1, 1 << (width - 1)
            declarations = ''.join(f'(declare-const {name} (_ BitVec {width}))'
                                   for name in ('a', 'b', 'c', 'd'))
            for operation in ('bvadd', 'bvmul'):
                formula = z3.parse_smt2_string(
                    declarations + f'(assert (= ({operation} a b c d) a))')[0]
                term = formula.arg(0)
                self.assertEqual(term.num_args(), 4)
                variables = term.children()
                for values in ((0, mask, high, 1), (mask, mask, mask, mask),
                               (high, 1, high, 1), (high - 1, 3 & mask, 2 & mask, mask)):
                    with self.subTest(width=width, operation=operation, values=values):
                        expected = (sum(values) if operation == 'bvadd' else
                                    values[0] * values[1] * values[2] * values[3]) & mask
                        self.concrete(term, list(zip(variables, values)), expected)

    def test_wrap_masks_and_shift_boundaries_match_modular_oracle(self):
        for width in (1, 4, 8, 32):
            mask, high = (1 << width) - 1, 1 << (width - 1)
            x = z3.BitVec('wrap_shift_input', width)
            low_mask = (1 << (width // 2)) - 1
            for value in (0, high, mask):
                terms = ((~x, (~value) & mask), (-x, (-value) & mask),
                         ((x + 1) & 0, 0), ((x + 1) & low_mask, (value + 1) & low_mask))
                for term, expected in terms:
                    with self.subTest(width=width, value=value, term=term):
                        self.concrete(term, [(x, value)], expected)
                for shift in (0, width - 1, width, mask):
                    with self.subTest(width=width, value=value, shift=shift):
                        self.concrete(x << shift, [(x, value)],
                                      0 if shift >= width else (value << shift) & mask)
                        self.concrete(z3.LShR(x, shift), [(x, value)],
                                      0 if shift >= width else value >> shift)

    def test_sign_extension_products_and_extracts_match_modular_oracle(self):
        for width in (1, 4, 8, 16, 32):
            mask, high = (1 << width) - 1, 1 << (width - 1)
            full_mask = (1 << (2 * width)) - 1
            a, b = z3.BitVecs('extended_a extended_b', width)
            product = z3.SignExt(width, a) * z3.SignExt(width, b)
            for av, bv in ((0, mask), (high, high - 1), (mask, mask), (mask, high)):
                signed_a = av - (1 << width) if av >= high else av
                signed_b = bv - (1 << width) if bv >= high else bv
                expected_product = (signed_a * signed_b) & full_mask
                terms = ((z3.SignExt(width, a), signed_a & full_mask),
                         (product, expected_product),
                         (z3.Extract(2 * width - 1, width, product), expected_product >> width),
                         (z3.ZeroExt(width, a) * z3.SignExt(width, b),
                          (av * signed_b) & full_mask))
                for term, expected in terms:
                    with self.subTest(width=width, av=av, bv=bv, term=term):
                        self.concrete(term, [(a, av), (b, bv)], expected)

    def test_unsupported_shifts_and_bitwise_forms_refuse(self):
        a, b = z3.BitVecs('unsupported_integer_a unsupported_integer_b', 8)
        terms = ((a << b, 'unknown'), (z3.LShR(a, b), 'unknown'), (a >> 1, 'unknown'),
                 (a ^ b, 'unknown'), (a | b, 'unknown'), (a & 0x55, 'sat'),
                 (a / b, 'unknown'), (z3.SRem(a, b), 'unknown'),
                 (z3.RotateLeft(a, 1), 'sat'))
        for term, rewritten_result in terms:
            with self.subTest(term=term):
                self.assertEqual(intenc.check([term == 1], 5, rewrite=False), 'unknown')
                with self.assertRaises(intenc.Unsupported):
                    intenc.Encoder(term.ctx).bv(term)
                # The trusted rewriter can express constant masks/rotations as
                # supported extracts and concatenations. That is an exact
                # scalar encoding, while the raw opcode still refuses.
                solver = z3.Solver()
                solver.set(timeout=5000)
                solver.add(term == 1)
                self.assertEqual(solver.check(), z3.sat, solver.reason_unknown())
                self.assertEqual(intenc.check([term == 1], 5), rewritten_result)


if __name__ == '__main__':
    verification_main()
