"""
LLVM IR Post-Processor for Y Compiler Output  (v5)
========================================================
Fixes call-site type mismatches by inserting conversion instructions.
Aliases are scoped per-function to avoid cross-function SSA conflicts.
"""
import re

def parse_func_sig(line):
    m = re.match(r'(?:declare|define)\s+(\S+)\s+@(\w+)\(([^)]*)\)', line)
    if not m: return None
    ret_ty = m.group(1)
    name   = m.group(2)
    pstr   = m.group(3).strip()
    if not pstr: return name, ret_ty, []
    params = []
    for p in pstr.split(','):
        p = p.strip()
        if p == '...': params.append('...')
        else: params.append(p.split()[0])
    return name, ret_ty, params

def split_args(s):
    args, depth, cur = [], 0, ''
    for ch in s:
        if ch in '({': depth += 1; cur += ch
        elif ch in ')}': depth -= 1; cur += ch
        elif ch == ',' and depth == 0: args.append(cur); cur = ''
        else: cur += ch
    if cur.strip(): args.append(cur)
    return args

INT_SIZES = {'i1': 1, 'i8': 8, 'i16': 16, 'i32': 32, 'i64': 64}

def is_struct_ty(ty):
    return ty.startswith('%')

def fix_args_with_conv(args_str, sig_params, ssa_types, indent, fresh_tmp):
    if not args_str.strip(): return args_str, []
    args = split_args(args_str)
    if len(args) != len(sig_params): return args_str, []
    new_args = []
    pre_instrs = []
    for arg, expected in zip(args, sig_params):
        parts = arg.strip().split(None, 1)
        if len(parts) != 2 or expected == '...':
            new_args.append(arg.strip())
            continue
        actual_ty, val = parts
        real_ty = ssa_types.get(val, actual_ty)
        if real_ty == expected:
            new_args.append(f'{expected} {val}')
        elif is_struct_ty(real_ty) or is_struct_ty(expected):
            # Struct types: pass through without conversion
            new_args.append(f'{expected} {val}')
        elif real_ty in INT_SIZES and expected in INT_SIZES:
            tmp = fresh_tmp()
            if INT_SIZES[real_ty] < INT_SIZES[expected]:
                pre_instrs.append(f'{indent}{tmp} = sext {real_ty} {val} to {expected}\n')
            else:
                pre_instrs.append(f'{indent}{tmp} = trunc {real_ty} {val} to {expected}\n')
            ssa_types[tmp] = expected
            new_args.append(f'{expected} {tmp}')
        elif real_ty == 'ptr' and expected in INT_SIZES:
            tmp = fresh_tmp()
            pre_instrs.append(f'{indent}{tmp} = ptrtoint ptr {val} to {expected}\n')
            ssa_types[tmp] = expected
            new_args.append(f'{expected} {tmp}')
        elif real_ty in INT_SIZES and expected == 'ptr':
            tmp = fresh_tmp()
            pre_instrs.append(f'{indent}{tmp} = inttoptr {real_ty} {val} to ptr\n')
            ssa_types[tmp] = 'ptr'
            new_args.append(f'ptr {tmp}')
        else:
            new_args.append(f'{expected} {val}')
    return ', '.join(new_args), pre_instrs

def main():
    with open('compiler.ll', 'r') as f:
        lines = f.readlines()

    # Pass 1: Collect function signatures
    sigs = {}
    for line in lines:
        s = line.strip()
        if s.startswith('declare ') or s.startswith('define '):
            r = parse_func_sig(s)
            if r: sigs[r[0]] = (r[1], r[2])

    # Pass 2: Process each function separately to scope aliases
    # First, split into function blocks
    func_blocks = []   # list of (start_line_idx, end_line_idx)
    current_func_start = None
    
    # We'll process line by line, collecting per-function aliases
    # and applying them within each function.
    
    ssa_types = {}
    fixed = []
    cur_ret = None
    tmp_counter = [0]
    func_aliases = {}  # per-function aliases
    func_start_idx = None  # index in 'fixed' where current function starts
    
    def fresh_tmp():
        tmp_counter[0] += 1
        return f'%__fix{tmp_counter[0]}'

    for line in lines:
        s = line.strip()

        if s.startswith('define '):
            # Apply aliases from previous function if any
            if func_aliases and func_start_idx is not None:
                apply_aliases(fixed, func_start_idx, func_aliases)
            
            r = parse_func_sig(s)
            if r: cur_ret = r[1]
            ssa_types.clear()
            tmp_counter[0] = 0
            func_aliases = {}
            func_start_idx = len(fixed)
        elif s == '}':
            # End of function - apply aliases
            fixed.append(line)
            if func_aliases and func_start_idx is not None:
                apply_aliases(fixed, func_start_idx, func_aliases)
            func_aliases = {}
            func_start_idx = None
            cur_ret = None
            continue

        # Track alloca - the SSA name IS a pointer (alloca returns ptr)
        m = re.match(r'\s+(%\w+)\s*=\s*alloca\s+(\S+)', line)
        if m: ssa_types[m.group(1)] = 'ptr'

        # Track load
        m = re.match(r'\s+(%\w+)\s*=\s*load\s+(\S+),', line)
        if m: ssa_types[m.group(1)] = m.group(2)

        # Track getelementptr
        m_gep = re.match(r'\s+(%\w+)\s*=\s*getelementptr\s+', line)
        if m_gep: ssa_types[m_gep.group(1)] = 'ptr'

        # Track bitcast / inttoptr / ptrtoint results
        m_cast = re.match(r'\s+(%\w+)\s*=\s*(?:bitcast|inttoptr|ptrtoint)\s+.*to\s+(\S+)', line)
        if m_cast: ssa_types[m_cast.group(1)] = m_cast.group(2)

        # Fix call with return value
        m_call = re.match(r'(\s+)(%\w+)\s*=\s*call\s+(\S+)\s+@(\w+)\((.*)\)(.*)', line)
        if m_call:
            indent, dest, call_ret, fname, args_str, trail = m_call.groups()
            if fname in sigs:
                sig_ret, sig_params = sigs[fname]
                if '...' not in sig_params:
                    args_str, pre_instrs = fix_args_with_conv(args_str, sig_params, ssa_types, indent, fresh_tmp)
                    for pi in pre_instrs:
                        fixed.append(pi)
                    call_ret = sig_ret
            ssa_types[dest] = call_ret
            if call_ret == 'void':
                fixed.append(f'{indent}call void @{fname}({args_str}){trail}\n')
            else:
                fixed.append(f'{indent}{dest} = call {call_ret} @{fname}({args_str}){trail}\n')
            continue

        # Fix void call
        m_call_v = re.match(r'(\s+)call\s+void\s+@(\w+)\((.*)\)(.*)', line)
        if m_call_v:
            indent, fname, args_str, trail = m_call_v.groups()
            if fname in sigs:
                sig_ret, sig_params = sigs[fname]
                if '...' not in sig_params:
                    args_str, pre_instrs = fix_args_with_conv(args_str, sig_params, ssa_types, indent, fresh_tmp)
                    for pi in pre_instrs:
                        fixed.append(pi)
            fixed.append(f'{indent}call void @{fname}({args_str}){trail}\n')
            continue

        # Elide inttoptr when source is already ptr
        m_itp = re.match(r'(\s+)(%\w+)\s*=\s*inttoptr\s+(\S+)\s+(%\w+)\s+to\s+(\S+)', line)
        if m_itp:
            indent, dest, src_ty, src_val, dst_ty = m_itp.groups()
            actual = ssa_types.get(src_val, src_ty)
            if actual == dst_ty:
                func_aliases[dest] = src_val
                ssa_types[dest] = dst_ty
                fixed.append(f'{indent}; elided inttoptr ({src_val} already {dst_ty})\n')
                continue
            ssa_types[dest] = dst_ty

        # Elide sext/trunc/zext when source already matches
        m_ext = re.match(r'(\s+)(%\w+)\s*=\s*(sext|trunc|zext)\s+(\S+)\s+(%\w+)\s+to\s+(\S+)', line)
        if m_ext:
            indent, dest, op, src_ty, src_val, dst_ty = m_ext.groups()
            actual = ssa_types.get(src_val, src_ty)
            if actual == dst_ty:
                func_aliases[dest] = src_val
                ssa_types[dest] = dst_ty
                fixed.append(f'{indent}; elided {op} ({src_val} already {dst_ty})\n')
                continue
            ssa_types[dest] = dst_ty

        # Fix ret
        m_ret = re.match(r'(\s+)ret\s+(\S+)\s+(.*)', line)
        if m_ret and cur_ret:
            indent, ret_ty, ret_val = m_ret.group(1), m_ret.group(2), m_ret.group(3).strip()
            if ret_ty != cur_ret and ret_ty != 'void':
                if cur_ret == 'void':
                    line = f'{indent}ret void\n'
                else:
                    line = f'{indent}ret {cur_ret} {ret_val}\n'

        fixed.append(line)

    with open('compiler_fixed.ll', 'w') as f:
        f.writelines(fixed)
    
    total_aliases = sum(1 for line in fixed if '; elided' in line)
    print(f"Fixed compiler.ll -> compiler_fixed.ll")
    print(f"  {len(sigs)} function signatures, {total_aliases} elided coercions")

def apply_aliases(lines, start_idx, aliases):
    """Apply alias substitutions within the function block [start_idx, len(lines))."""
    def resolve(name):
        visited = set()
        while name in aliases and name not in visited:
            visited.add(name)
            name = aliases[name]
        return name
    
    sorted_aliases = sorted(aliases.keys(), key=len, reverse=True)
    for i in range(start_idx, len(lines)):
        line = lines[i]
        if '; elided' in line:
            continue
        for alias in sorted_aliases:
            if alias in line:
                resolved = resolve(alias)
                line = re.sub(re.escape(alias) + r'(?!\w)', resolved, line)
        lines[i] = line

if __name__ == '__main__':
    main()
