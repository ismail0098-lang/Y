import re

def fix():
    with open('compiler.c', 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Fix multi-line strings in ystr_new
    def fix_strings(match):
        s = match.group(1)
        s = s.replace('\r\n', '\\n').replace('\n', '\\n')
        return 'ystr_new("' + s + '")'
    content = re.sub(r'ystr_new\("((?:[^"\\]|\\.)*?)"\)', fix_strings, content, flags=re.DOTALL)

    # 2. Fix main return type
    content = content.replace('void main(void)', 'int main(void)')

    # 3. Extract types and protos from User Code
    user_header = '/* ── User Code ───────────────────────────────── */'
    if user_header not in content: return
    parts = content.split(user_header)
    runtime = parts[0]
    user_code = parts[1]

    # Extract all components
    enums = re.findall(r'typedef enum \{.*?\} \w+;', user_code, flags=re.DOTALL)
    for e in enums: user_code = user_code.replace(e, '')
    
    struct_defs = re.findall(r'typedef struct \{.*?\} \w+;', user_code, flags=re.DOTALL)
    for s in struct_defs: user_code = user_code.replace(s, '')
    
    structs = re.findall(r'struct \w+ \{.*?\};', user_code, flags=re.DOTALL)
    for s in structs: user_code = user_code.replace(s, '')
    
    fwd_decls = re.findall(r'typedef struct \w+ \w+;', user_code)
    for d in fwd_decls: user_code = user_code.replace(d, '')
    
    macros = re.findall(r'#define \w+ \(\(TokenKind\)\{.*?\}\)', user_code)
    for m in macros: user_code = user_code.replace(m, '')
    
    # Protos: lines that look like 'Type Name(Args);'
    protos = re.findall(r'\n\w+ \w+\(.*? \w+.*?\);', user_code)
    for p in protos: user_code = user_code.replace(p, '')

    # Re-insert in correct order for C
    # 1. Forward decls
    # 2. Enums
    # 3. Structs (the component ones first, then the main ones)
    # 4. Macros
    # 5. Prototypes
    
    # We'll just put enums before everything else.
    ordered_types = "\n\n".join(fwd_decls + enums + struct_defs + structs + macros + protos)
    
    content = runtime + user_header + "\n\n" + ordered_types + "\n\n" + user_code

    with open('compiler_fixed.c', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Fixed compiler.c -> compiler_fixed.c")

if __name__ == "__main__":
    fix()
