import re
import string

def normalize_text(text: str) -> str:
    """Standardizes text casing and whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def tokenize_words(text: str) -> list[str]:
    """Tokenizes text into words stripping punctuation."""
    norm = normalize_text(text)
    # Remove punctuation
    clean = norm.translate(str.maketrans('', '', string.punctuation))
    tokens = clean.split()
    return tokens

def compute_word_jaccard(text1: str, text2: str) -> float:
    """Computes Jaccard similarity over word token sets."""
    set1 = set(tokenize_words(text1))
    set2 = set(tokenize_words(text2))
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return float(intersection / union) if union > 0 else 0.0

def compute_char_ngram_jaccard(text1: str, text2: str, n: int = 3) -> float:
    """Computes Jaccard similarity over character n-grams."""
    t1 = normalize_text(text1)
    t2 = normalize_text(text2)
    ngrams1 = set([t1[i:i+n] for i in range(len(t1) - n + 1)]) if len(t1) >= n else set([t1])
    ngrams2 = set([t2[i:i+n] for i in range(len(t2) - n + 1)]) if len(t2) >= n else set([t2])
    if not ngrams1 or not ngrams2:
        return 0.0
    intersection = len(ngrams1.intersection(ngrams2))
    union = len(ngrams1.union(ngrams2))
    return float(intersection / union) if union > 0 else 0.0

