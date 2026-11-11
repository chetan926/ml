import re
from collections import Counter

JAVA_KEYWORDS = {
    'abstract', 'assert', 'boolean', 'break', 'byte', 'case', 'catch', 'char', 'class',
    'const', 'continue', 'default', 'do', 'double', 'else', 'enum', 'extends', 'final',
    'finally', 'float', 'for', 'goto', 'if', 'implements', 'import', 'instanceof', 'int',
    'interface', 'long', 'native', 'new', 'package', 'private', 'protected', 'public',
    'return', 'short', 'static', 'strictfp', 'super', 'switch', 'synchronized', 'this',
    'throw', 'throws', 'transient', 'try', 'void', 'volatile', 'while', 'true', 'false', 'null'
}

JAVA_OPERATORS = [
    '==', '!=', '<=', '>=', '&&', '||', '++', '--', '+=', '-=', '*=', '/=', '%=',
    '+', '-', '*', '/', '%', '<', '>', '=', '!', '&', '|', '^', '~', '?', ':'
]

def remove_comments(code: str) -> str:
    """Removes single-line and multi-line comments from Java code."""
    # Remove block comments /* ... */
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    # Remove line comments // ...
    code = re.sub(r'//.*', '', code)
    return code

def extract_tokens(code: str) -> list[str]:
    """Tokenizes Java code into alphanumeric identifiers/keywords, operators, and symbols."""
    clean = remove_comments(code)
    # Match keywords/identifiers, numbers, multi-char operators, or single symbols
    pattern = r'[a-zA-Z_]\w*|\d+|==|!=|<=|>=|&&|\|\||\+\+|--|[+\-*/%=<>!&|^~?:;,.(){}\[\]]'
    tokens = re.findall(pattern, clean)
    return tokens

def anonymize_identifiers(code: str) -> str:
    """
    Replaces custom student identifiers with placeholder tokens (VAR_1, VAR_2, FUNC_1),
    preserving Java keywords, as proposed in Gandhi et al. (2024).
    """
    tokens = extract_tokens(code)
    id_map = {}
    anon_tokens = []
    var_idx = 1

    for tok in tokens:
        if tok in JAVA_KEYWORDS:
            anon_tokens.append(tok)
        elif re.match(r'^[a-zA-Z_]\w*$', tok):
            if tok not in id_map:
                id_map[tok] = f"ID_{var_idx}"
                var_idx += 1
            anon_tokens.append(id_map[tok])
        else:
            anon_tokens.append(tok)
    return ' '.join(anon_tokens)

def extract_code_features(code: str) -> dict:
    """
    Extracts structural, lexical, and syntactic features from Java code submissions:
    - Token count & keyword frequency distribution
    - Cyclomatic complexity proxy (control branching points)
    - Line counts (raw vs stripped)
    - Operator frequency
    """
    raw_lines = len(code.splitlines())
    clean_code = remove_comments(code)
    clean_lines = len([l for l in clean_code.splitlines() if l.strip()])
    tokens = extract_tokens(clean_code)

    keywords = [t for t in tokens if t in JAVA_KEYWORDS]
    operators = [t for t in tokens if t in JAVA_OPERATORS]
    identifiers = [t for t in tokens if re.match(r'^[a-zA-Z_]\w*$', t) and t not in JAVA_KEYWORDS]

    # Cyclomatic proxy
    branch_tokens = {'if', 'for', 'while', 'case', 'catch', '&&', '||', '?'}
    branch_count = sum(1 for t in tokens if t in branch_tokens)

    return {
        'raw_lines': raw_lines,
        'clean_lines': clean_lines,
        'total_tokens': len(tokens),
        'keyword_count': len(keywords),
        'operator_count': len(operators),
        'identifier_count': len(identifiers),
        'unique_identifiers': len(set(identifiers)),
        'cyclomatic_proxy': branch_count + 1,
        'token_list': tokens,
        'clean_code': clean_code
    }

