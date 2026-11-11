"""
Reproducible Experiment Configuration and Global Parameters (Phase 3.1)
Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection
"""

from pathlib import Path
import os
import sys
import random
import numpy as np

# -----------------------------------------------------------------------------
# 1. Directory Structure and Absolute Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
BASELINE_SRC_DIR = PROJECT_ROOT / "src"
RESEARCH_UPGRADE_DIR = PROJECT_ROOT / "research_upgrade"
RESEARCH_UPGRADE_SRC_DIR = RESEARCH_UPGRADE_DIR / "src"
RESEARCH_UPGRADE_DOCS_DIR = RESEARCH_UPGRADE_DIR / "documentation"
DATASETS_DIR = PROJECT_ROOT / "datasets"
PROCESSED_DATA_DIR = DATASETS_DIR / "processed"
IR_PLAG_DIR = DATASETS_DIR / "raw" / "ir_plag" / "extracted" / "IR-Plag-Dataset"

# Dataset Files
TEXT_DATASET_PATH = PROCESSED_DATA_DIR / "text_plagiarism_benchmark.csv"
CODE_DATASET_PATH = PROCESSED_DATA_DIR / "code_plagiarism_benchmark.csv"

# Phase 2 Audit Directory (Source of Truth for Partitions & Baselines)
PHASE2_AUDIT_DIR = RESEARCH_UPGRADE_DIR / "audits" / "phase_2_final"
PHASE2_MANIFEST_PATH = PHASE2_AUDIT_DIR / "repository_sha256_manifest.csv"
PHASE2_TEXT_LEAKAGE_PATH = PHASE2_AUDIT_DIR / "text_group_leakage_report.json"
PHASE2_TEXT_EXCLUSION_PATH = PHASE2_AUDIT_DIR / "text_exclusion_ledger.csv"
PHASE2_CODE_DUPLICATE_PATH = PHASE2_AUDIT_DIR / "code_duplicate_accounting.json"
PHASE2_CASE05_FEASIBILITY_PATH = PHASE2_AUDIT_DIR / "case_05_fold_feasibility_report.json"

# Phase 3 Artifacts Directory
PHASE3_ARTIFACTS_DIR = RESEARCH_UPGRADE_DIR / "artifacts" / "phase_3"
PHASE3_RUNS_DIR = PHASE3_ARTIFACTS_DIR / "runs"
PHASE3_LOGS_DIR = PHASE3_ARTIFACTS_DIR / "logs"

# Ensure runtime directories exist
PHASE3_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
PHASE3_RUNS_DIR.mkdir(parents=True, exist_ok=True)
PHASE3_LOGS_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# 2. Deterministic Reproducibility & Random Seeds
# -----------------------------------------------------------------------------
RANDOM_SEED: int = 42

def set_global_seed(seed: int = RANDOM_SEED) -> None:
    """
    Sets deterministic seeds across Python standard library, NumPy, and PyTorch (if available).
    Ensures CPU determinism without violating platform boundaries.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except (ImportError, OSError):
        pass


# -----------------------------------------------------------------------------
# 3. Dataset Accounting & Exact Partition Targets
# -----------------------------------------------------------------------------
# Text Modality Partitions (from 19,666 raw rows; 8 purged; 19,658 clean rows)
N_RAW_TEXT_ROWS: int = 19666
N_PURGED_CROSS_CONTAMINATION: int = 6
N_PURGED_REDUNDANT_DUPLICATES: int = 2
N_TOTAL_PURGED_ROWS: int = 8
N_TOTAL_CLEAN_ROWS: int = 19658

# Development Pool Allocations (from 9,837 clean val rows)
N_DEV_POOL: int = 9837
N_TRAIN: int = 4000         # 1,349 unique premises -> linear fusion w*
N_VAL: int = 1500           #   508 unique premises -> thresholds (theta_low, theta_high)
N_CAL_PROB: int = 1000      #   336 unique premises -> Platt scaling (A*, B*)
N_CAL_CONF: int = 1000      #   334 unique premises -> conformal quantile q_hat
N_DEV_RESERVE: int = 2337   #   791 unique premises -> held-out reserve

# Test Pool Allocations (from 9,821 clean test rows)
N_TEST_POOL: int = 9821
N_TEST: int = 3000          # 1,015 unique premises -> UNTOUCHED TEST BENCHMARK
N_TEST_RESERVE: int = 6821  # 2,307 unique premises -> held-out reserve

# Code Modality Parameters (from 733 raw pairs; 24 rows in 11 dupe groups)
N_RAW_CODE_PAIRS: int = 733
N_CODE_DUPLICATE_ROWS: int = 24
N_CODE_DUPLICATE_GROUPS: int = 11
N_CODE_POLICY_A_RETAINED: int = 720  # Drop subsequent occurrences
N_CODE_POLICY_B_RETAINED: int = 709  # Drop entire collision groups
N_CODE_REVERSED_DUPLICATES: int = 0
N_IR_PLAG_PHYSICAL_JAVA_FILES: int = 467

# Case-05 Cross-Fitting Proof Constants
CASE_05_PHYSICAL_JAVA_FILES: int = 69
CASE_05_TOTAL_PAIRS: int = 107
CASE_05_POSITIVE_PAIRS: int = 53
CASE_05_NEGATIVE_PAIRS: int = 54
CASE_05_BASE_PROGRAMS_COUNT: int = 16


# -----------------------------------------------------------------------------
# 4. Uncertainty, Conformal Prediction & Calibration Parameters
# -----------------------------------------------------------------------------
# Conformal Significance Level (95% target coverage)
DEFAULT_CONFORMAL_EPSILON: float = 0.05
MIN_CALIBRATION_GROUPS: int = 19  # ceil((1 - eps) / eps)

# Probability Calibration Evaluation Diagnostics
ECE_NUM_BINS: int = 10
NLL_CLIP_EPS: float = 1e-15

# Multi-Objective Pareto Search Bounds
MIN_VAL_RECALL: float = 0.95
DESIGN_TARGET_RECALL: float = 0.98
MAX_WORKLOAD: float = 0.15
PROJECTED_ONLINE_LATENCY_MS: float = 1.1375
STAGE1_LATENCY_TARGET_MS: float = 0.050


# -----------------------------------------------------------------------------
# 5. Operational 4-State Routing Definitions
# -----------------------------------------------------------------------------
ROUTING_STATES = {
    "LOW_RISK_REVIEW_DEFERRED": {
        "description": "Pair determined low risk by Stage-1 screening or Stage-2 conformal confident negative",
        "human_review_required": False,
        "is_audited": False
    },
    "HIGH_RISK_DIRECT_BYPASS": {
        "description": "Pair exceeds Stage-1 high threshold; routed directly to human audit without Stage 2",
        "human_review_required": True,
        "is_audited": True
    },
    "HIGH_RISK_CANDIDATE_FLAGGED": {
        "description": "Pair confirmed plagiarized by Stage-2 cross-encoder and conforming prediction set {1}",
        "human_review_required": True,
        "is_audited": True
    },
    "AMBIGUOUS_HUMAN_REVIEW": {
        "description": "Pair is uncertain with prediction set {0, 1} or falls into intermediate probability band",
        "human_review_required": True,
        "is_audited": True
    },
    "ABSTAIN_UNRESOLVED": {
        "description": "Empty prediction set [] signifying predictive ambiguity; routed to expert auditor queue",
        "human_review_required": True,
        "is_audited": True
    }
}
