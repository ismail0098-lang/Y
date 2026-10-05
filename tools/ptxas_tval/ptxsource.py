"""Read normalized PTX in the validator's explicit line-oriented subset.

All specification-side scanners use this reader: an inline comment must not
hide an instruction, and a commented declaration must not alter the signature
or control-flow subject. Quoted directive strings are preserved verbatim.

The executors consume one instruction per line. A directive may share a line
with executable PTX, which their old ``startswith('.')`` checks discarded.
``read`` therefore accepts only an explicit grammar of effect-free directive
lines. This is a trusted parser for that subset, not a general PTX lexer;
unknown directives and mixed directive/instruction lines refuse. Pure repeated
declarations are split so every declaration also reaches the layout scanner.
``strip_comments`` remains independent for token-based proof-subject identity.
"""
import re


_TOKENS = re.compile(r'"(?:\\.|[^"\\])*"|//[^\n]*|/\*[\s\S]*?\*/|/\*')

_NAME = r'[A-Za-z_$][\w$]*'
_TYPE = r'\.(?:pred|[bsuf](?:8|16|32|64))'
_STRING = r'"(?:\\.|[^"\\])*"'
_PARAM = (r'\.param\s+(?:\.align\s+\d+\s+)?' + _TYPE
          + r'(?:\s+\.ptr(?:\s+\.(?:global|shared|local|const))?'
            r'(?:\s+\.align\s+\d+)?)?\s+' + _NAME + r'(?:\[\d*\])?')
_PARAMS = _PARAM + r'(?:\s*,\s*' + _PARAM + r')*\s*,?'
_REG = r'\.reg\s+' + _TYPE + r'\s+%?' + _NAME + r'(?:<\d+>)?(?:\s*,\s*%?' + _NAME + r'(?:<\d+>)?)*'
_SHARED = r'(?:\.extern\s+)?\.shared\s+\.align\s+\d+\s+\.b(?:8|16|32|64)\s+' + _NAME + r'\[\d*\]'
_DECLARATION = re.compile(r'(?:' + _REG + r'|' + _SHARED + r')')
_LABEL = re.compile(r'\$?' + _NAME + r'\s*:')
_DIRECTIVES = [re.compile(pattern) for pattern in (
    r'\.version\s+\d+\.\d+',
    r'\.target\s+sm_\d+[af]?',
    r'\.address_size\s+(?:32|64)',
    r'(?:\.(?:visible|weak|extern)\s+)*\.entry\s+' + _NAME
        + r'\s*\(\s*(?:' + _PARAMS + r')?\s*\)?',
    _PARAMS + r'\s*\)?',
    r'\.(?:maxnreg|minnctapersm|maxnctapersm)\s+\d+',
    r'\.(?:maxntid|reqntid)\s+\d+(?:\s*,\s*\d+){0,2}',
    r'\.file\s+\d+\s+' + _STRING,
    r'\.loc\s+\d+\s+\d+\s+\d+',
    r'\.pragma\s+' + _STRING + r'\s*;',
)]


def require_directive_lines(text):
    """Check ignored directives and structural lines in a source snapshot.

    Return an equivalent snapshot with pure repeated declarations on separate
    lines. Full-line matching, including the payload after each semicolon,
    prevents a store, predicate update, branch or brace from being discarded.
    Declaration initializers, functions, debug sections and richer metadata
    forms are outside this subset and must be added with explicit semantics.
    Nested lexical scopes refuse because the register model has one flat name
    dictionary. Loop nesting expressed by labels introduces no lexical scope.
    """
    output, depth = [], 0
    for number, line in enumerate(text.splitlines(), 1):
        source = line.strip()
        if not source.startswith('.'):
            if not source:
                output.append(line)
                continue
            if source == '{':
                if depth:
                    raise Exception('UNMODELLED PTX nested lexical scope: the register '
                                    'model does not preserve shadowed declarations '
                                    '(refusing, not guessing)')
                depth += 1
            elif source == '}':
                depth -= 1
                if depth < 0:
                    raise Exception('UNMODELLED PTX unmatched closing brace '
                                    '(refusing, not guessing)')
            elif source == ')' and not depth:
                pass  # Close a multiline entry parameter list, with no payload.
            elif not depth or source.startswith(('{', '}')) or _LABEL.match(source):
                if not depth or not _LABEL.fullmatch(source):
                    raise Exception(f'UNMODELLED PTX structural line {number}: {source!r}; '
                                    'inline or unparsed executable payload '
                                    '(refusing, not guessing)')
            elif not source.endswith(';') or source.count(';') != 1:
                raise Exception(f'UNMODELLED PTX source line {source!r}; '
                                'expected one complete instruction per line '
                                '(refusing, not guessing)')
            output.append(line)
            continue
        if any(pattern.fullmatch(source) for pattern in _DIRECTIVES):
            output.append(line)
            continue
        if source.endswith(';'):
            declarations = source[:-1].split(';')
            if all(_DECLARATION.fullmatch(part.strip()) for part in declarations):
                output.extend(part.strip() + ';' for part in declarations)
                continue
        raise Exception(f'UNMODELLED PTX directive line {number}: {source!r}; '
                        'unknown directive or mixed directive/executable payload '
                        '(refusing, not guessing)')
    if depth:
        raise Exception('UNMODELLED PTX unmatched opening brace '
                        '(refusing, not guessing)')
    return '\n'.join(output) + ('\n' if text.endswith('\n') else '')


def strip_comments(text):
    """Normalize an already-read source snapshot using the same lexer as read."""
    def replace(match):
        token = match.group()
        if token.startswith('"'):
            return token
        if token == '/*':
            raise Exception('UNMODELLED PTX unterminated block comment '
                            '(refusing, not guessing)')
        # Spaces keep `mov/* comment */.u32` from becoming `mov.u32`, while
        # retaining newlines preserves line-oriented directives and diagnostics.
        return ''.join('\n' if ch == '\n' else ' ' for ch in token)

    return _TOKENS.sub(replace, text)


def read(path):
    with open(path, encoding='utf-8') as source:
        return require_directive_lines(strip_comments(source.read()))
