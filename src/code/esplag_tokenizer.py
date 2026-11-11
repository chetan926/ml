"""
ES-Plag Java Source Code Tokenizer.
Based on Sulistiani & Karnalim (2019), "ES-Plag: Efficient and sensitive source code
plagiarism detection tool for academic environment", Comput. Appl. Eng. Educ.
DOI: 10.1002/cae.22066.

Features:
- Robust comment removal that preserves string and character literals.
- Java token extraction preserving sequential ordering for RKRGST.
- Token frequency extraction for vector indexing.
"""

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

# Regex to match string literals, char literals, block comments, or line comments
COMMENT_LITERAL_REGEX = re.compile(
    r'("(?:\\.|[^"\\])*")|'          # Group 1: Double-quoted string literal
    r"('(?:\\.|[^'\\])*')|"          # Group 2: Single-quoted character literal
    r'(/\*[\s\S]*?\*/)|'            # Group 3: Multi-line block comment
    r'(//[^\r\n]*)'                  # Group 4: Single-line comment
)

def remove_comments_safe(code: str) -> str:
    """
    Strips single-line and multi-line comments from Java source code
    without modifying string or character literals.
    """
    def replacer(match):
        # If matched group 1 (string) or group 2 (char), keep it unchanged
        if match.group(1):
            return match.group(1)
        if match.group(2):
            return match.group(2)
        # Otherwise it was a comment, remove it
        return " "

    return COMMENT_LITERAL_REGEX.sub(replacer, code)

# Regex pattern for Java tokens
TOKEN_PATTERN = re.compile(
    r'(?:[a-zA-Z_]\w*)'                                # Identifiers and keywords
    r'|(?:"(?:\\.|[^"\\])*")'                          # String literals
    r"|(?:'(?:\\.|[^'\\])*')"                          # Char literals
    r'|(?:\d+(?:\.\d+)?(?:[eE][+-]?\d+)?[fFdDlL]?)'    # Numeric literals
    r'|(?:>>>|<<|>>|==|!=|<=|>=|&&|\|\||\+\+|--|\+=|-=|\*=|/=|%=|&=|\|=|\^=)' # Multi-char operators
    r'|(?:[+\-*/%=<>!&|^~?:;,.(){}\[\]])'              # Single-char operators/delimiters
)

def tokenize_code(code: str, lower: bool = False) -> list[str]:
    """
    Tokenizes Java source code into an ordered list of tokens suitable for RKRGST.
    Comments are removed safely prior to tokenization.
    """
    clean_code = remove_comments_safe(code)
    if lower:
        clean_code = clean_code.lower()
    tokens = TOKEN_PATTERN.findall(clean_code)
    return tokens

def get_token_frequency(tokens: list[str]) -> Counter:
    """
    Returns a Counter mapping tokens to their occurrence frequencies.
    """
    return Counter(tokens)

