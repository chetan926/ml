import os
import sys
import time
import subprocess
import pandas as pd
import numpy as np

# Ensure offline execution and suppress cache warnings
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
mpl_dir = os.path.abspath('.matplotlib')
os.makedirs(mpl_dir, exist_ok=True)
os.environ['MPLCONFIGDIR'] = mpl_dir

# Ensure src is in python path
sys.path.insert(0, os.path.abspath('src'))

from data_loader import prepare_text_dataset, prepare_code_dataset, compute_dataset_statistics
from create_literature_docs import create_literature
from evaluation import run_text_experiments, run_code_experiments
from experiment_evaluator import (
    run_esplag_experiments,
    run_setu_hybrid_experiments,
    run_recall_constrained_comparison
)
from qualitative_analysis import run_qualitative_text_analysis, run_qualitative_code_analysis
from visualization import generate_all_visualizations

RESULTS_DIR = 'experiments/results'
FIG_DIR = 'experiments/figures'

def main():
    print("="*75)
    print("TYPED-DECISION PRE-FILTERING FOR COST-EFFICIENT ACADEMIC INTEGRITY DETECTION")
    print("Master Experimental Pipeline Reproduction Script (11 Comprehensive Phases)")
    print("="*75)
    t_start = time.time()

    # -------------------------------------------------------------
    # PHASE 1: Literature Documentation and Methodology Audit
    # -------------------------------------------------------------
    print("\n[PHASE 1/11] Literature Documentation and Reference Methodology Audit...")
    create_literature()

    # -------------------------------------------------------------
    # PHASE 2: Dataset Loading, Preprocessing and Validation Verification
    # -------------------------------------------------------------
    print("\n[PHASE 2/11] Preparing and Preprocessing Datasets...")
    prepare_text_dataset(sample_size=None)
    prepare_code_dataset()
    compute_dataset_statistics()

    # -------------------------------------------------------------
    # PHASE 3: Baseline Text Plagiarism Experiments
    # -------------------------------------------------------------
    print("\n[PHASE 3/11] Executing Baseline Text Plagiarism Experiments (TF-IDF, SBERT, Conditions A, B, C)...")
    text_results, text_context = run_text_experiments(sample_size=3000)

    # -------------------------------------------------------------
    # PHASE 4: Baseline Source Code Plagiarism Experiments
    # -------------------------------------------------------------
    print("\n[PHASE 4/11] Executing Baseline Source Code Experiments (Token Jaccard, SVM, RF, Conditions A, B, C)...")
    code_results, code_context = run_code_experiments()

    # -------------------------------------------------------------
    # PHASE 5: ES-Plag Source Code Plagiarism Experiments (Experiment D)
    # -------------------------------------------------------------
    print("\n[PHASE 5/11] Executing ES-Plag Code Experiments (Dynamic Cosine Filtering + RKRGST + Penalty)...")
    esplag_results = run_esplag_experiments(code_context['df_val'], code_context['df_test'])

    # -------------------------------------------------------------
    # PHASE 6: Setu Hybrid Text Plagiarism Experiments (Experiment E)
    # -------------------------------------------------------------
    print("\n[PHASE 6/11] Executing Setu Hybrid Text Experiments (E1 Full Hybrid vs. E2 Staged Hybrid)...")
    test_ids = text_context['df_test']['id'].tolist() if 'id' in text_context['df_test'].columns else None
    setu_results = run_setu_hybrid_experiments(
        text_context['val_ref'], text_context['val_sus'], text_context['y_val'],
        text_context['test_ref'], text_context['test_sus'], text_context['y_test'],
        test_ids=test_ids
    )

    # -------------------------------------------------------------
    # PHASE 7: Recall-Constrained Multi-Method Comparison & Random Control
    # -------------------------------------------------------------
    print("\n[PHASE 7/11] Evaluating Recall-Constrained Optimization & Random Controls (Targets >= 0.95 and >= 0.98)...")
    multi_comparison_df = run_recall_constrained_comparison(text_context, code_context, targets=[0.95, 0.98])

    # -------------------------------------------------------------
    # PHASE 8: Qualitative Error Analysis Across Modalities
    # -------------------------------------------------------------
    print("\n[PHASE 8/11] Running Qualitative Error Analysis Across Text and Code Modalities...")
    run_qualitative_text_analysis()
    run_qualitative_code_analysis()

    # -------------------------------------------------------------
    # PHASE 9: High-Resolution Publication Figure Generation (Figures 01 to 17)
    # -------------------------------------------------------------
    print("\n[PHASE 9/11] Generating All 17 High-Resolution Publication Figures...")
    generate_all_visualizations()

    # -------------------------------------------------------------
    # PHASE 10: Master Results Synthesis & Verification Check
    # -------------------------------------------------------------
    print("\n[PHASE 10/11] Performing Master Results Synthesis & Verification Check...")
    required_csvs = [
        'text_experiments_summary.csv',
        'code_experiments_summary.csv',
        'code_obfuscation_level_recall.csv',
        'threshold_sensitivity_data.csv',
        'qualitative_text_cases.csv',
        'qualitative_code_cases.csv',
        'esplag_code_filtering_summary.csv',
        'esplag_code_pair_results.csv',
        'esplag_code_confusion_matrix.csv',
        'esplag_code_threshold_sensitivity.csv',
        'esplag_code_obfuscation_analysis.csv',
        'setu_hybrid_text_summary.csv',
        'setu_hybrid_text_pair_results.csv',
        'setu_hybrid_confusion_matrix.csv',
        'setu_hybrid_threshold_sensitivity.csv',
        'setu_hybrid_routing_analysis.csv',
        'recall_constrained_multi_method_comparison.csv'
    ]

    all_csvs_ok = True
    for fname in required_csvs:
        p = os.path.join(RESULTS_DIR, fname)
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            print(f"  [ERROR] Missing or empty CSV artifact: {p}")
            all_csvs_ok = False
        else:
            print(f"  [OK] Verified CSV artifact: {fname} ({os.path.getsize(p)} bytes)")

    required_figs = [f"{i:02d}_" for i in range(1, 18)]
    all_figs_ok = True
    fig_files = os.listdir(FIG_DIR) if os.path.exists(FIG_DIR) else []
    for prefix in required_figs:
        matched = [f for f in fig_files if f.startswith(prefix) and f.endswith('.png')]
        if not matched:
            print(f"  [ERROR] Missing figure with prefix: {prefix}")
            all_figs_ok = False
        else:
            p = os.path.join(FIG_DIR, matched[0])
            print(f"  [OK] Verified Figure: {matched[0]} ({os.path.getsize(p)} bytes)")

    if all_csvs_ok and all_figs_ok:
        print("\n  >> Master Artifact Verification Check: PASS (All CSVs and Figures Confirmed)")
    else:
        print("\n  >> Master Artifact Verification Check: INCOMPLETE ARTIFACTS DETECTED")

    # -------------------------------------------------------------
    # PHASE 11: Academic Research Report & PDF Generation
    # -------------------------------------------------------------
    print("\n[PHASE 11/11] Compiling Academic Research PDF Report...")
    pdf_script = os.path.abspath('scratch/generate_pdf_report.py')
    if os.path.exists(pdf_script):
        ret = subprocess.run([sys.executable, pdf_script], capture_output=True, text=True)
        if ret.returncode == 0:
            print("  [OK] Academic Research PDF Report successfully generated at reports/final_research_report.pdf")
        else:
            print(f"  [WARNING] PDF compilation output:\n{ret.stderr}")

    total_time = time.time() - t_start
    print("\n" + "="*75)
    print(f"MASTER EXPERIMENTAL PIPELINE EXECUTED SUCCESSFULLY IN {total_time:.2f} SECONDS!")
    print("="*75)

if __name__ == '__main__':
    main()
