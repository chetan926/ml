"""
Complexity and Operator Feature Extractor (Phase 3.2, Code View 4)
Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection
"""

import re
from typing import Dict, Any, List
import numpy as np

from src.code.esplag_tokenizer import remove_comments_safe

OPERATORS = [
    r"\+\+", r"--", r"\+=", r"-=", r"\*=", r"/=", r"%=",
    r"==", r"!=", r"<=", r">=", r"&&", r"\|\|",
    r"\+", r"-", r"\*", r"/", r"%", r"<", r">", r"=", r"!"
]

CONTROL_KEYWORDS = [
    r"\bif\b", r"\belse\b", r"\bfor\b", r"\bwhile\b", r"\bswitch\b",
    r"\bcase\b", r"\bdefault\b", r"\breturn\b", r"\btry\b", r"\bcatch\b",
    r"\bfinally\b", r"\bthrow\b", r"\bbreak\b", r"\bcontinue\b"
]


def count_patterns(code: str, patterns: List[str]) -> np.ndarray:
    """Counts occurrences of each regex pattern in code string."""
    counts = np.array([len(re.findall(p, code)) for p in patterns], dtype=np.float64)
    return counts


def vector_cosine_sim(v1: np.ndarray, v2: np.ndarray) -> float:
    """Computes cosine similarity between two 1D non-negative vectors."""
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 == 0.0 or n2 == 0.0:
        return 1.0 if n1 == n2 else 0.0
    sim = float(np.dot(v1, v2) / (n1 * n2))
    return float(max(0.0, min(1.0, sim)))


def count_sloc(code: str) -> int:
    """Counts non-blank source lines of code."""
    lines = [line.strip() for line in code.splitlines()]
    return sum(1 for line in lines if line)


def estimate_cyclomatic_complexity(code: str) -> int:
    """Estimates McCabe cyclomatic complexity: decision points + 1."""
    decision_patterns = [
        r"\bif\b", r"\bwhile\b", r"\bfor\b", r"\bcase\b", r"&&", r"\|\|"
    ]
    decisions = sum(len(re.findall(p, code)) for p in decision_patterns)
    return decisions + 1


class CodeComplexityExtractor:
    """
    Extracts software engineering complexity and operator distribution features:
    1. Operator frequency distribution cosine similarity.
    2. Control-flow keyword distribution cosine similarity.
    3. Approximate McCabe cyclomatic complexity ratio.
    4. Source Lines of Code (SLOC) ratio.
    """

    def __init__(self):
        pass

    def extract(self, ref_code: str, sus_code: str) -> Dict[str, float]:
        """Extracts complexity features for a Java code pair."""
        ref_src = remove_comments_safe(str(ref_code)) if ref_code is not None else ""
        sus_src = remove_comments_safe(str(sus_code)) if sus_code is not None else ""

        if not ref_src.strip() or not sus_src.strip():
            return {
                "code_operator_dist_sim": 0.0,
                "code_control_flow_sim": 0.0,
                "code_cyclomatic_sim": 0.0,
                "code_sloc_ratio": 0.0
            }

        # 1. Operator Distribution Similarity
        op_ref = count_patterns(ref_src, OPERATORS)
        op_sus = count_patterns(sus_src, OPERATORS)
        op_sim = vector_cosine_sim(op_ref, op_sus)

        # 2. Control Flow Distribution Similarity
        ctrl_ref = count_patterns(ref_src, CONTROL_KEYWORDS)
        ctrl_sus = count_patterns(sus_src, CONTROL_KEYWORDS)
        ctrl_sim = vector_cosine_sim(ctrl_ref, ctrl_sus)

        # 3. Cyclomatic Complexity Similarity
        cc_ref = estimate_cyclomatic_complexity(ref_src)
        cc_sus = estimate_cyclomatic_complexity(sus_src)
        cc_sim = float(min(cc_ref, cc_sus) / max(cc_ref, cc_sus))

        # 4. Source Lines of Code (SLOC) Ratio
        sloc_ref = count_sloc(ref_src)
        sloc_sus = count_sloc(sus_src)
        max_sloc = max(sloc_ref, sloc_sus)
        sloc_ratio = float(min(sloc_ref, sloc_sus) / max_sloc) if max_sloc > 0 else 1.0

        return {
            "code_operator_dist_sim": round(op_sim, 6),
            "code_control_flow_sim": round(ctrl_sim, 6),
            "code_cyclomatic_sim": round(cc_sim, 6),
            "code_sloc_ratio": round(sloc_ratio, 6)
        }

    @property
    def feature_names(self) -> List[str]:
        return [
            "code_operator_dist_sim",
            "code_control_flow_sim",
            "code_cyclomatic_sim",
            "code_sloc_ratio"
        ]

