"""
Phase 5 Adversarial Robustness Suite.

Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for
Cost-Efficient Academic Integrity Detection — Q1 Upgrade (Phase 5).

Addresses the Phase 4.1 audit gap: "no paraphrase/AI-rewriting attack evaluation".

Because the audit environment is network-isolated (no LLM API access), the
paraphrase robustness study implements a *simulated* adversarial paraphrase
protocol composed of deterministic, seeded text transformations motivated by
documented LLM-rewriting side effects:

    T1  SynonymSubstitutionAttack   — lexical drift (WordNet-style curated map)
    T2  LocalWordOrderAttack        — local constituent reordering
    T3  CharacterNoiseAttack        — misspelling / keyboard-transposition typos
    T4  FillerInjectionAttack       — verbosity inflation ("it is important...")
    T5  CombinedParaphraseAttack    — T1+T2+T3+T4 composed

For source code, IR-Plag already embodies the Faidhi–Robinson L1–L6 obfuscation
spectrum; Phase 5 adds an *extra-benchmark* attack not present in IR-Plag:

    C1  SystematicIdentifierRenaming — consistent identifier/method renaming
        (the single most common LLM code-rewriting side effect).

All attacks are applied to the SUSPICIOUS side of positive (plagiarized) pairs
to measure recall degradation, with overall metrics recomputed on the attacked
admixture (attacked positives + original negatives).
"""

import re
from typing import Any, Dict, List, Tuple

import numpy as np

# Curated high-frequency English synonym map (WordNet-style), sufficient to
# produce lexical drift in the SNLI-derived sentence corpus.
SYNONYM_MAP: Dict[str, str] = {
    "a": "one", "two": "a pair of", "three": "a trio of", "people": "persons",
    "man": "male", "woman": "female", "men": "males", "women": "females",
    "boy": "lad", "girl": "lass", "child": "kid", "children": "kids",
    "dog": "canine", "cat": "feline", "is": "appears to be", "are": "seem to be",
    "big": "large", "small": "little", "huge": "enormous", "tiny": "minute",
    "fast": "quick", "slow": "sluggish", "happy": "glad", "sad": "unhappy",
    "angry": "upset", "walk": "stroll", "walking": "strolling", "run": "jog",
    "running": "jogging", "jump": "leap", "jumping": "leaping", "sit": "rest",
    "sitting": "resting", "stand": "be upright", "standing": "waiting upright",
    "sleep": "doze", "sleeping": "dozing", "eat": "dine", "eating": "dining",
    "drink": "sip", "drinking": "sipping", "look": "gaze", "looking": "gazing",
    "watch": "observe", "watching": "observing", "play": "engage", "playing": "engaging",
    "hold": "grasp", "holding": "grasping", "carry": "bear", "carrying": "bearing",
    "wear": "don", "wearing": "donning", "red": "crimson", "blue": "azure",
    "green": "verdant", "white": "ivory", "black": "ebony", "yellow": "golden",
    "brown": "tawny", "pink": "rosy", "gray": "grey", "orange": "amber",
    "shirt": "top", "pants": "trousers", "hat": "cap", "jacket": "coat",
    "dress": "gown", "shoe": "footwear", "shoes": "footwear",
    "street": "road", "road": "street", "park": "playground", "field": "meadow",
    "water": "aqua", "ocean": "sea", "sea": "ocean", "beach": "shore",
    "snow": "snowfall", "rain": "drizzle", "sun": "sunshine",
    "outside": "outdoors", "inside": "indoors", "near": "close to", "next": "adjacent",
    "in": "within", "on": "upon", "at": "by", "with": "alongside",
    "the": "the", "this": "the", "that": "the", "there": "over there",
    "here": "at this spot", "now": "currently", "today": "this day",
    "some": "several", "many": "numerous", "few": "a handful of",
    "very": "quite", "really": "truly", "also": "as well", "too": "as well",
    "and": "plus", "but": "however", "so": "therefore", "because": "since",
    "good": "fine", "nice": "pleasant", "new": "fresh", "old": "aged",
    "young": "youthful", "beautiful": "pretty", "pretty": "lovely",
    "group": "crowd", "crowd": "gathering", "friend": "companion",
    "friends": "companions", "family": "relatives", "couple": "pair",
    "picture": "photo", "photo": "image", "camera": "photographic device",
    "phone": "mobile", "computer": "machine", "table": "desk",
    "chair": "seat", "window": "pane", "door": "entrance", "car": "vehicle",
    "bike": "bicycle", "bicycle": "cycle", "motorcycle": "motorbike",
    "bus": "coach", "train": "locomotive", "ball": "sphere", "game": "match",
    "food": "meal", "sandwich": "snack", "smile": "grin", "smiling": "grinning",
    "laugh": "chuckle", "laughing": "chuckling", "talk": "converse",
    "talking": "conversing", "speak": "utter", "read": "peruse",
    "reading": "perusing", "write": "jot", "writing": "jotting",
    "dance": "dance about", "dancing": "dancing about", "swim": "take a dip",
    "swimming": "taking a dip", "ride": "ride on", "riding": "riding on",
    "climb": "scale", "climbing": "scaling", "build": "construct",
    "building": "constructing", "make": "create", "making": "creating",
    "take": "grab", "taking": "grabbing", "give": "hand", "giving": "handing",
    "use": "utilize", "using": "utilizing", "show": "display",
    "showing": "displaying", "buy": "purchase", "selling": "vending",
    "begin": "commence", "end": "conclude", "start": "commence",
    "stop": "cease", "help": "assist", "helping": "assisting",
    "work": "labor", "working": "laboring", "rest": "relax", "resting": "relaxing",
}

FILLER_PHRASES = [
    "It is important to note that",
    "As a matter of fact,",
    "In this particular instance,",
    "Without a doubt,",
    "Interestingly enough,",
]

_TOKEN_RE = re.compile(r"\b\w+\b")
_JAVA_IDENT_RE = re.compile(r"\b[a-zA-Z_$][a-zA-Z0-9_$]*\b")
_JAVA_KEYWORDS = {
    "public", "private", "protected", "static", "final", "void", "int", "long",
    "double", "float", "boolean", "char", "byte", "short", "class", "interface",
    "enum", "extends", "implements", "return", "if", "else", "for", "while", "do",
    "switch", "case", "break", "continue", "new", "try", "catch", "finally",
    "throw", "throws", "import", "package", "this", "super", "instanceof",
    "abstract", "synchronized", "volatile", "transient", "native", "strictfp",
    "assert", "default", "goto", "const", "null", "true", "false", "String",
    "System", "out", "println", "print", "main", "Override", "Deprecated",
    "SuppressWarnings", "java", "util", "io", "lang", "Scanner", "Math",
}


def synonym_substitution(text: str, rate: float, rng: np.random.RandomState) -> str:
    """T1: replaces words with curated synonyms at probability `rate`."""
    def _sub(match: re.Match) -> str:
        word = match.group(0)
        low = word.lower()
        if low in SYNONYM_MAP and rng.rand() < rate:
            rep = SYNONYM_MAP[low]
            return rep.capitalize() if word[0].isupper() else rep
        return word
    return _TOKEN_RE.sub(_sub, text)


def local_word_order_shuffle(text: str, window: int = 3, rate: float = 0.5,
                             rng: np.random.RandomState = None) -> str:
    """T2: swaps adjacent tokens inside non-overlapping windows of size `window`."""
    tokens = _TOKEN_RE.findall(text)
    if len(tokens) < window:
        return text
    i = 0
    while i + window - 1 < len(tokens):
        if rng.rand() < rate:
            mid = i + window // 2
            tokens[i], tokens[mid] = tokens[mid], tokens[i]
        i += window
    # Rebuild text by permuting matched tokens in-place (count is invariant).
    it = iter(tokens)
    return _TOKEN_RE.sub(lambda m: next(it), text)


def character_noise(text: str, rate: float, rng: np.random.RandomState) -> str:
    """T3: adjacent-character transpositions at rate `rate` per word character."""
    def _noise_word(word: str) -> str:
        chars = list(word)
        for i in range(len(chars) - 1):
            if rng.rand() < rate:
                chars[i], chars[i + 1] = chars[i + 1], chars[i]
        return "".join(chars)
    return _TOKEN_RE.sub(lambda m: _noise_word(m.group(0)), text)


def filler_injection(text: str, rng: np.random.RandomState) -> str:
    """T4: injects ONE filler phrase (prepended or mid-sentence), mimicking
    single-insertion introductory-phrase habits of LLM paraphrasing."""
    tokens = text.split()
    if len(tokens) < 8:
        return f"{rng.choice(FILLER_PHRASES)} {text}"
    if rng.rand() < 0.5:
        return f"{rng.choice(FILLER_PHRASES)} {text}"
    mid = max(2, len(tokens) // 3)
    return " ".join(tokens[:mid]) + "," + f"\x20{rng.choice(FILLER_PHRASES).lower()} " + " ".join(tokens[mid:])


def combined_paraphrase(text: str, rng: np.random.RandomState) -> str:
    """T5: realistic composite paraphrase = synonym(0.25) + local order (0.35) +
    a single filler injection. (Char-noise is a separate stress axis, not part of
    the realistic composite.)"""
    t = synonym_substitution(text, 0.25, rng)
    t = local_word_order_shuffle(t, window=3, rate=0.35, rng=rng)
    t = filler_injection(t, rng)
    return t


def systematic_identifier_renaming(code: str) -> str:
    """
    C1 (code): consistent, deterministic identifier renaming.
    Maps each non-keyword Java identifier to a stable obfuscated name:
    class-like -> Cnn, method-like -> mnn, variable-like -> vnn.
    """
    mapping: Dict[str, str] = {}
    counters = {"v": 0, "m": 0, "c": 0}

    def _rename(match: re.Match) -> str:
        ident = match.group(0)
        if ident in _JAVA_KEYWORDS or len(ident) <= 1:
            return ident
        if ident in mapping:
            return mapping[ident]
        if ident[0].isupper():
            kind = "c"
        else:
            kind = "m" if ident[0].islower() and ident[1:2].isupper() else "v"
        counters[kind] += 1
        mapping[ident] = f"{kind}{counters[kind]:02d}"
        return mapping[ident]

    return _JAVA_IDENT_RE.sub(_rename, code)


def apply_text_attack(
    suspicious_texts: List[str],
    attack: str,
    seed: int,
    **kwargs,
) -> List[str]:
    """Applies a named attack to a list of suspicious texts (deterministic given seed)."""
    attacked = []
    for i, txt in enumerate(suspicious_texts):
        rng = np.random.RandomState(seed + i)
        t = str(txt)
        if attack == "synonym_light":
            t = synonym_substitution(t, 0.15, rng)
        elif attack == "synonym_heavy":
            t = synonym_substitution(t, 0.30, rng)
        elif attack == "word_order":
            t = local_word_order_shuffle(t, window=3, rate=0.40, rng=rng)
        elif attack == "char_noise":
            t = character_noise(t, 0.03, rng)
        elif attack == "filler":
            t = filler_injection(t, rng)
        elif attack == "combined":
            t = combined_paraphrase(t, rng)
        else:
            raise ValueError(f"Unknown attack type: {attack}")
        attacked.append(t)
    return attacked
