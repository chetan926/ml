import os
import sys

mpl_dir = os.path.abspath('.matplotlib')
os.makedirs(mpl_dir, exist_ok=True)
os.environ['MPLCONFIGDIR'] = mpl_dir

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

FIG_DIR = 'experiments/figures'
RESULTS_DIR = 'experiments/results'

def generate_all_visualizations():
    os.makedirs(FIG_DIR, exist_ok=True)
    df_text = pd.read_csv(os.path.join(RESULTS_DIR, 'text_experiments_summary.csv'))
    df_code = pd.read_csv(os.path.join(RESULTS_DIR, 'code_experiments_summary.csv'))
    df_obf = pd.read_csv(os.path.join(RESULTS_DIR, 'code_obfuscation_level_recall.csv'))

    palette = ['#2b5c8f', '#d95f02', '#7570b3', '#e7298a', '#1b9e77']

    # -------------------------------------------------------------
    # 1. Accuracy Comparison Across Methods and Operating Conditions
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 5), dpi=300)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # Text Accuracy
    ax1 = axes[0]
    bars1 = ax1.barh(df_text['Method / Condition'], df_text['Accuracy'], color=palette[:len(df_text)])
    ax1.set_xlim(0, 1.05)
    ax1.set_title('Text Plagiarism Detection Accuracy\n(MIT / SNLI Benchmark)', fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel('Classification Accuracy', fontsize=10)
    for bar in bars1:
        w = bar.get_width()
        ax1.text(w + 0.015, bar.get_y() + bar.get_height()/2, f"{w:.3f}", va='center', fontsize=9, fontweight='bold')

    # Code Accuracy
    ax2 = axes[1]
    bars2 = ax2.barh(df_code['Method / Condition'], df_code['Accuracy'], color=palette[:len(df_code)])
    ax2.set_xlim(0, 1.05)
    ax2.set_title('Java Code Plagiarism Detection Accuracy\n(IR-Plag Benchmark)', fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel('Classification Accuracy', fontsize=10)
    for bar in bars2:
        w = bar.get_width()
        ax2.text(w + 0.015, bar.get_y() + bar.get_height()/2, f"{w:.3f}", va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    p1 = os.path.join(FIG_DIR, '01_accuracy_comparison.png')
    plt.savefig(p1, bbox_inches='tight')
    plt.close()
    print(f"Saved {p1}")

    # -------------------------------------------------------------
    # 2. Precision, Recall and F1-Score Comparison
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Text Metrics
    ax1 = axes[0]
    text_melt = df_text.melt(id_vars=['Method / Condition'], value_vars=['Precision', 'Recall', 'F1'], var_name='Metric', value_name='Score')
    sns.barplot(data=text_melt, x='Score', y='Method / Condition', hue='Metric', ax=ax1, palette='Blues_d')
    ax1.set_xlim(0, 1.1)
    ax1.set_title('Text Benchmark: Precision, Recall & F1', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right')

    # Code Metrics
    ax2 = axes[1]
    code_melt = df_code.melt(id_vars=['Method / Condition'], value_vars=['Precision', 'Recall', 'F1'], var_name='Metric', value_name='Score')
    sns.barplot(data=code_melt, x='Score', y='Method / Condition', hue='Metric', ax=ax2, palette='Greens_d')
    ax2.set_xlim(0, 1.1)
    ax2.set_title('Source Code Benchmark: Precision, Recall & F1', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right')

    plt.tight_layout()
    p2 = os.path.join(FIG_DIR, '02_precision_recall_f1_comparison.png')
    plt.savefig(p2, bbox_inches='tight')
    plt.close()
    print(f"Saved {p2}")

    # -------------------------------------------------------------
    # 3. Confusion Matrices
    # -------------------------------------------------------------
    with open(os.path.join(RESULTS_DIR, 'text_results_detailed.json')) as f:
        t_det = json.load(f)
    with open(os.path.join(RESULTS_DIR, 'code_results_detailed.json')) as f:
        c_det = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), dpi=300)

    cm_text_tfidf = np.array(t_det['res_tfidf']['confusion_matrix'])
    cm_text_sbert = np.array(t_det['res_sbert']['confusion_matrix'])
    cm_code_svm = np.array(c_det['res_svm']['confusion_matrix'])
    cm_code_pipe = np.array(c_det['res_code_c']['confusion_matrix'])

    matrices = [
        (cm_text_tfidf, 'Text: Method 1 (TF-IDF Lexical)', axes[0, 0], 'Blues'),
        (cm_text_sbert, 'Text: Method 2 (Contextual SBERT)', axes[0, 1], 'Purples'),
        (cm_code_svm, 'Code: Method 3 (Syntactic SVM)', axes[1, 0], 'Greens'),
        (cm_code_pipe, 'Code: Condition C (Typed-Decision Pipeline)', axes[1, 1], 'Oranges')
    ]

    for cm, title, ax, cmap in matrices:
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, cbar=False, ax=ax,
                    xticklabels=['Non-Plag', 'Plagiarized'], yticklabels=['Non-Plag', 'Plagiarized'])
        ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('Ground Truth Label')

    plt.tight_layout()
    p3 = os.path.join(FIG_DIR, '03_confusion_matrices.png')
    plt.savefig(p3, bbox_inches='tight')
    plt.close()
    print(f"Saved {p3}")

    # -------------------------------------------------------------
    # 4. Runtime & Latency Comparison
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    ax1 = axes[0]
    bars = ax1.bar(df_text['Method / Condition'], df_text['Avg Latency (ms)'], color=['#3182bd', '#e6550d', '#de2d26', '#fd8d3c', '#31a354'])
    ax1.set_ylabel('Average Latency per Pair (ms, log scale)')
    ax1.set_yscale('log')
    ax1.set_title('Text Processing Latency\n(Logarithmic Scale)', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='x', rotation=30)
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h * 1.2, f"{h:.2f}ms", ha='center', fontsize=8, fontweight='bold')

    ax2 = axes[1]
    bars2 = ax2.bar(df_code['Method / Condition'], df_code['Avg Latency (ms)'], color=['#74c476', '#31a354', '#006d2c', '#9ecae1', '#fd8d3c'])
    ax2.set_ylabel('Average Latency per Pair (ms)')
    ax2.set_title('Source Code Processing Latency', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=30)
    for bar in bars2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 0.05, f"{h:.3f}ms", ha='center', fontsize=8, fontweight='bold')

    plt.tight_layout()
    p4 = os.path.join(FIG_DIR, '04_runtime_latency_comparison.png')
    plt.savefig(p4, bbox_inches='tight')
    plt.close()
    print(f"Saved {p4}")

    # -------------------------------------------------------------
    # 5. Candidate Recall vs. Pruning Ratio Trade-off
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 5.5), dpi=300)
    cond_labels = ['Condition A\n(Full Analysis)', 'Condition B\n(Fixed Threshold)', 'Condition C\n(Typed Pre-Filter)']
    text_conds = df_text[df_text['Method / Condition'].str.startswith(('A:', 'B:', 'C:'))]

    plt.plot(text_conds['Pruning Ratio (%)'], text_conds['Candidate Recall'] * 100, marker='o', markersize=9, linewidth=2.5, color='#2b5c8f', label='Text Benchmark (SBERT Pipeline)')
    code_conds = df_code[df_code['Method / Condition'].str.startswith(('A:', 'B:', 'C:'))]
    plt.plot(code_conds['Pruning Ratio (%)'], code_conds['Candidate Recall'] * 100, marker='s', markersize=9, linewidth=2.5, color='#2ca02c', label='Code Benchmark (SVM Pipeline)')

    for _, row in text_conds.iterrows():
        plt.annotate(f"{row['Method / Condition'].split(':')[0]}: {row['Candidate Recall']*100:.1f}%\n(Prune {row['Pruning Ratio (%)']}%)",
                     (row['Pruning Ratio (%)'], row['Candidate Recall']*100),
                     textcoords="offset points", xytext=(-15, 12), ha='center', fontsize=8, fontweight='bold')

    plt.axhline(98.0, color='red', linestyle='--', alpha=0.7, label='Safety Target (>= 98% Candidate Recall)')
    plt.title('Candidate Recall Preservation vs. Pruning Ratio', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Pruning Ratio (% of Document Pairs Bypassing Expensive Model)', fontsize=11)
    plt.ylabel('Candidate Recall (% True Plagiarism Retained)', fontsize=11)
    plt.ylim(50, 105)
    plt.xlim(-5, 95)
    plt.legend(loc='lower left', frameon=True)
    plt.tight_layout()
    p5 = os.path.join(FIG_DIR, '05_candidate_recall_vs_pruning_ratio.png')
    plt.savefig(p5, bbox_inches='tight')
    plt.close()
    print(f"Saved {p5}")

    # -------------------------------------------------------------
    # 6. Expensive Detector Call Rate
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 5), dpi=300)
    categories = ['Text: Condition A\n(Full Analysis)', 'Text: Condition B\n(Fixed 50%)', 'Text: Condition C\n(Proposed Typed)',
                  'Code: Condition A\n(Full Analysis)', 'Code: Condition B\n(Fixed 50%)', 'Code: Condition C\n(Proposed Typed)']
    call_rates = [
        float(text_conds[text_conds['Method / Condition'].str.startswith('A')]['Expensive Call Rate (%)'].iloc[0]),
        float(text_conds[text_conds['Method / Condition'].str.startswith('B')]['Expensive Call Rate (%)'].iloc[0]),
        float(text_conds[text_conds['Method / Condition'].str.startswith('C')]['Expensive Call Rate (%)'].iloc[0]),
        float(code_conds[code_conds['Method / Condition'].str.startswith('A')]['Expensive Call Rate (%)'].iloc[0]),
        float(code_conds[code_conds['Method / Condition'].str.startswith('B')]['Expensive Call Rate (%)'].iloc[0]),
        float(code_conds[code_conds['Method / Condition'].str.startswith('C')]['Expensive Call Rate (%)'].iloc[0]),
    ]
    colors = ['#de2d26', '#fc9272', '#31a354', '#de2d26', '#fc9272', '#31a354']
    bars = plt.bar(categories, call_rates, color=colors, width=0.6, edgecolor='#555555')
    plt.ylabel('Expensive Detector Invocations (% of Total Pairs)', fontsize=11)
    plt.title('Computational Workload Reduction: Expensive Detector Call Rate', fontsize=13, fontweight='bold', pad=12)
    plt.ylim(0, 115)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, h + 2.5, f"{h:.1f}%", ha='center', fontsize=9, fontweight='bold')
    plt.axhline(100.0, color='gray', linestyle=':', alpha=0.5)
    plt.tight_layout()
    p6 = os.path.join(FIG_DIR, '06_expensive_detector_call_rate.png')
    plt.savefig(p6, bbox_inches='tight')
    plt.close()
    print(f"Saved {p6}")

    # -------------------------------------------------------------
    # 7. Recall vs. Computational Cost Trade-off
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 5.5), dpi=300)
    for df_sub, mod_name, color, marker in [(text_conds, 'Text', '#1f77b4', 'o'), (code_conds, 'Code', '#2ca02c', 's')]:
        plt.scatter(df_sub['Avg Latency (ms)'], df_sub['Recall'] * 100, s=150, color=color, marker=marker, label=f"{mod_name} Pipeline", zorder=3)
        for _, row in df_sub.iterrows():
            cond_code = row['Method / Condition'].split(':')[0]
            plt.annotate(f"{cond_code} ({row['Avg Latency (ms)']:.1f}ms, {row['Recall']*100:.1f}%)",
                         (row['Avg Latency (ms)'], row['Recall']*100),
                         textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8, fontweight='bold')

    plt.xlabel('Average Latency per Pair (ms)', fontsize=11)
    plt.ylabel('Detection Recall (%)', fontsize=11)
    plt.title('Detection Recall vs. Computational Cost Pareto Trade-Off', fontsize=13, fontweight='bold', pad=12)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    p7 = os.path.join(FIG_DIR, '07_recall_vs_cost_tradeoff.png')
    plt.savefig(p7, bbox_inches='tight')
    plt.close()
    print(f"Saved {p7}")

    # -------------------------------------------------------------
    # 8. Threshold Sensitivity Curves
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 5), dpi=300)
    df_sens = pd.read_csv(os.path.join(RESULTS_DIR, 'threshold_sensitivity_data.csv'))
    thetas = df_sens['Threshold'].values
    cand_recalls = df_sens['Candidate_Recall'].values * 100.0
    prune_ratios = df_sens['Pruning_Ratio'].values * 100.0
    calibrated_th = float(df_sens['Calibrated_Theta_Low'].iloc[0])

    plt.plot(thetas, cand_recalls, label='Empirical Candidate Recall (%)', color='#2b5c8f', linewidth=2.5)
    plt.plot(thetas, prune_ratios, label='Empirical Pruning Ratio (%)', color='#d95f02', linewidth=2.5, linestyle='--')
    plt.axvline(calibrated_th, color='green', linestyle=':', label=f'Calibrated Text theta_low ({calibrated_th:.4f})', linewidth=2)
    plt.title('Validation Threshold Sensitivity: Low-Risk Boundary (theta_low)\n(Authentic Validation Empirical Sweep)', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Candidate Pre-Filter Decision Threshold (theta_low)', fontsize=11)
    plt.ylabel('Percentage (%)', fontsize=11)
    plt.legend(loc='center right')
    plt.tight_layout()
    p8 = os.path.join(FIG_DIR, '08_threshold_sensitivity_curves.png')
    plt.savefig(p8, bbox_inches='tight')
    plt.close()
    print(f"Saved {p8}")

    # -------------------------------------------------------------
    # 9. Source Code Obfuscation Resilience across Levels L1-L6
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 5), dpi=300)
    bars = plt.bar(df_obf['Obfuscation Level'], df_obf['SVM Recall'] * 100,
                   color=['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02'],
                   edgecolor='#333333', width=0.55)
    plt.ylim(0, 115)
    plt.ylabel('Plagiarism Detection Recall (%)', fontsize=11)
    plt.xlabel('Faidhi-Robinson Obfuscation Level', fontsize=11)
    plt.title('Method 3 (SVM) Obfuscation Resilience Across Attack Levels\n(IR-Plag Java Corpus)', fontsize=13, fontweight='bold', pad=12)

    for bar, desc in zip(bars, df_obf['Description']):
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, h + 2.5, f"{h:.1f}%", ha='center', fontsize=9, fontweight='bold')
        # Annotate short description below x-axis
    plt.tight_layout()
    p9 = os.path.join(FIG_DIR, '09_code_obfuscation_levels.png')
    plt.savefig(p9, bbox_inches='tight')
    plt.close()
    print(f"Saved {p9}")

    # Generate ES-Plag and Setu Hybrid visualizations (Figures 10-17)
    generate_esplag_visualizations()
    generate_setu_visualizations()

    print("\nAll 17 publication figures successfully generated!")

def generate_esplag_visualizations():
    """Generates Figures 10, 11, 12, 13 for ES-Plag Code Methodology."""
    # -------------------------------------------------------------
    # 10. ES-Plag Runtime and Latency Comparison
    # -------------------------------------------------------------
    sum_path = os.path.join(RESULTS_DIR, 'esplag_code_filtering_summary.csv')
    if os.path.exists(sum_path):
        df_es = pd.read_csv(sum_path)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

        # Total Runtime
        ax1 = axes[0]
        bars1 = ax1.barh(df_es['Configuration'], df_es['Total_Runtime_Sec'], color=['#7570b3', '#386cb0', '#fdb462', '#2ca02c'])
        ax1.set_xlabel('Total Runtime (Seconds)', fontsize=11)
        ax1.set_title('Total Test Evaluation Runtime\n(IR-Plag Java Corpus)', fontsize=12, fontweight='bold')
        for bar in bars1:
            w = bar.get_width()
            ax1.text(w + 0.05, bar.get_y() + bar.get_height()/2, f"{w:.2f}s", va='center', fontsize=9, fontweight='bold')

        # Average Latency per Pair
        ax2 = axes[1]
        bars2 = ax2.barh(df_es['Configuration'], df_es['Avg_Latency_ms'], color=['#7570b3', '#386cb0', '#fdb462', '#2ca02c'])
        ax2.set_xlabel('Average Latency per Pair (ms)', fontsize=11)
        ax2.set_title('Per-Pair Latency Comparison\n(Pre-Filtering vs Full RKRGST)', fontsize=12, fontweight='bold')
        for bar in bars2:
            w = bar.get_width()
            ax2.text(w + 0.1, bar.get_y() + bar.get_height()/2, f"{w:.2f}ms", va='center', fontsize=9, fontweight='bold')

        plt.tight_layout()
        p10 = os.path.join(FIG_DIR, '10_esplag_runtime_comparison.png')
        plt.savefig(p10, bbox_inches='tight')
        plt.close()
        print(f"Saved {p10}")

    # -------------------------------------------------------------
    # 11. ES-Plag Dynamic Filter Threshold Sensitivity (Recall vs. Pruning)
    # -------------------------------------------------------------
    sens_path = os.path.join(RESULTS_DIR, 'esplag_code_threshold_sensitivity.csv')
    if os.path.exists(sens_path):
        df_sens = pd.read_csv(sens_path)
        plt.figure(figsize=(9, 5), dpi=300)
        plt.plot(df_sens['initial_threshold_ratio'], df_sens['candidate_recall'] * 100.0,
                 marker='o', linewidth=2.5, color='#2b5c8f', label='Candidate Recall (%)')
        plt.plot(df_sens['initial_threshold_ratio'], df_sens['pruning_ratio'] * 100.0,
                 marker='s', linewidth=2.5, color='#d95f02', linestyle='--', label='Pruning Ratio (%)')
        plt.plot(df_sens['initial_threshold_ratio'], df_sens['expensive_call_rate'] * 100.0,
                 marker='^', linewidth=2.0, color='#1b9e77', linestyle=':', label='Expensive GST Call Rate (%)')

        plt.axvline(0.50, color='gray', linestyle='-.', alpha=0.8, label='Reference initial_threshold (0.50)')
        plt.xlabel('Initial Threshold Proportional Ratio (initial_threshold)', fontsize=11)
        plt.ylabel('Percentage (%)', fontsize=11)
        plt.title('ES-Plag Dynamic Cosine Pre-Filtering Sensitivity\n(Validation Set Optimization)', fontsize=13, fontweight='bold', pad=12)
        plt.legend(loc='center right')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        p11 = os.path.join(FIG_DIR, '11_esplag_recall_vs_pruning.png')
        plt.savefig(p11, bbox_inches='tight')
        plt.close()
        print(f"Saved {p11}")

    # -------------------------------------------------------------
    # 12. ES-Plag Penalty Normalization Analysis
    # -------------------------------------------------------------
    pair_path = os.path.join(RESULTS_DIR, 'esplag_code_pair_results.csv')
    if os.path.exists(pair_path):
        df_pairs = pd.read_csv(pair_path)
        plag_pairs = df_pairs[(df_pairs['ground_truth'] == 1) & (df_pairs['obfuscation_level'] != 'None')].copy()
        if len(plag_pairs) > 0:
            plt.figure(figsize=(10, 5.5), dpi=300)
            avg_by_level = plag_pairs.groupby('obfuscation_level')[['penalized_average', 'penalized_max', 'raw_gst_average']].mean()

            x = np.arange(len(avg_by_level))
            width = 0.25

            plt.bar(x - width, avg_by_level['raw_gst_average'], width=width, label='Raw GST (No Penalty)', color='#386cb0')
            plt.bar(x, avg_by_level['penalized_average'], width=width, label='Penalized Average (ES-Plag)', color='#2ca02c')
            plt.bar(x + width, avg_by_level['penalized_max'], width=width, label='Penalized Max', color='#fdb462')

            plt.xticks(x, avg_by_level.index, fontsize=10, fontweight='bold')
            plt.xlabel('IR-Plag Obfuscation Level', fontsize=11)
            plt.ylabel('Mean Similarity Score', fontsize=11)
            plt.title('Impact of ES-Plag Penalty Normalization on Obfuscated Code\n(Fragmenation Penalty Deducts (num_subsequences - 1))', fontsize=13, fontweight='bold', pad=12)
            plt.legend(loc='lower left')
            plt.grid(True, axis='y', linestyle='--', alpha=0.6)
            plt.tight_layout()
            p12 = os.path.join(FIG_DIR, '12_esplag_penalty_analysis.png')
            plt.savefig(p12, bbox_inches='tight')
            plt.close()
            print(f"Saved {p12}")

    # -------------------------------------------------------------
    # 13. ES-Plag Obfuscation-Level Recall Comparison
    # -------------------------------------------------------------
    obf_path = os.path.join(RESULTS_DIR, 'esplag_code_obfuscation_analysis.csv')
    if os.path.exists(obf_path):
        df_obf_es = pd.read_csv(obf_path)
        plt.figure(figsize=(10, 5.5), dpi=300)
        x = np.arange(len(df_obf_es))
        width = 0.20

        plt.bar(x - 1.5*width, df_obf_es['ESPlag_No_Filter_Recall'] * 100.0, width=width, label='Full RKRGST Baseline', color='#7570b3')
        plt.bar(x - 0.5*width, df_obf_es['ESPlag_Penalized_Avg_Recall'] * 100.0, width=width, label='ES-Plag Penalized Average', color='#2ca02c')
        plt.bar(x + 0.5*width, df_obf_es['ESPlag_Penalized_Max_Recall'] * 100.0, width=width, label='ES-Plag Penalized Max', color='#fdb462')
        plt.bar(x + 1.5*width, df_obf_es['ESPlag_Raw_GST_Recall'] * 100.0, width=width, label='ES-Plag Raw GST', color='#386cb0')

        plt.xticks(x, df_obf_es['Obfuscation_Level'], fontsize=10, fontweight='bold')
        plt.xlabel('IR-Plag Obfuscation Level (L1 to L6)', fontsize=11)
        plt.ylabel('Plagiarism Detection Recall (%)', fontsize=11)
        plt.title('ES-Plag Obfuscation-Level Recall Comparison\n(Pre-Filtering vs Full RKRGST across Transformations)', fontsize=13, fontweight='bold', pad=12)
        plt.legend(loc='lower left')
        plt.grid(True, axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        p13 = os.path.join(FIG_DIR, '13_esplag_obfuscation_comparison.png')
        plt.savefig(p13, bbox_inches='tight')
        plt.close()
        print(f"Saved {p13}")

def generate_setu_visualizations():
    """Generates Figures 14, 15, 16, 17 for Setu Hybrid Text Methodology."""
    # -------------------------------------------------------------
    # 14. Setu Hybrid Performance Metrics Comparison
    # -------------------------------------------------------------
    setu_sum_path = os.path.join(RESULTS_DIR, 'setu_hybrid_text_summary.csv')
    if os.path.exists(setu_sum_path):
        df_setu = pd.read_csv(setu_sum_path)
        plt.figure(figsize=(10, 5.5), dpi=300)
        melted = df_setu.melt(id_vars=['Condition'], value_vars=['Accuracy', 'Precision', 'Recall', 'F1_Score'],
                              var_name='Metric', value_name='Score')
        sns.barplot(data=melted, x='Condition', y='Score', hue='Metric', palette='Set2')
        plt.xticks(rotation=15, ha='right', fontsize=9, fontweight='bold')
        plt.ylim(0, 1.1)
        plt.ylabel('Evaluation Score', fontsize=11)
        plt.title('Setu Text Hybrid Methodology Comparison\n(E1 Full Hybrid vs. E2 Cost-Efficient Staged Hybrid)', fontsize=13, fontweight='bold', pad=12)
        plt.legend(loc='lower right')
        plt.tight_layout()
        p14 = os.path.join(FIG_DIR, '14_setu_hybrid_comparison.png')
        plt.savefig(p14, bbox_inches='tight')
        plt.close()
        print(f"Saved {p14}")

    # -------------------------------------------------------------
    # 15. Lexical TF-IDF vs. SBERT Semantic Score Distribution
    # -------------------------------------------------------------
    pair_path = os.path.join(RESULTS_DIR, 'setu_hybrid_text_pair_results.csv')
    if os.path.exists(pair_path):
        df_pairs = pd.read_csv(pair_path)
        # Sample if large for clean scatter rendering
        if len(df_pairs) > 1500:
            df_plot = df_pairs.sample(n=1500, random_state=42)
        else:
            df_plot = df_pairs

        plt.figure(figsize=(9, 6), dpi=300)
        gt_col = 'ground_truth' if 'ground_truth' in df_plot.columns else 'y_true'
        pos = df_plot[df_plot[gt_col] == 1]
        neg = df_plot[df_plot[gt_col] == 0]

        plt.scatter(neg['tfidf_score'], neg['sbert_score'], color='#2b5c8f', alpha=0.35, s=25, label='Non-Plagiarized', zorder=2)
        plt.scatter(pos['tfidf_score'], pos['sbert_score'], color='#d95f02', alpha=0.55, s=35, label='Plagiarized', zorder=3)

        # Plot staged hybrid threshold lines
        plt.axvline(0.05, color='gray', linestyle='--', linewidth=1.5, label=r'Pruning Boundary ($\theta_{low}=0.05$)')
        plt.axvline(0.80, color='purple', linestyle=':', linewidth=1.5, label=r'Direct-Flag Boundary ($\theta_{high}=0.80$)')

        plt.xlabel('Lexical TF-IDF Cosine Similarity', fontsize=11)
        plt.ylabel('SBERT Semantic Cosine Similarity', fontsize=11)
        plt.title('Lexical vs. Semantic Similarity Space in Text Detection\n(Illustrating Direct Copies vs. Paraphrastic Plagiarism)', fontsize=13, fontweight='bold', pad=12)
        plt.legend(loc='lower right', framealpha=0.9)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        p15 = os.path.join(FIG_DIR, '15_text_lexical_vs_semantic.png')
        plt.savefig(p15, bbox_inches='tight')
        plt.close()
        print(f"Saved {p15}")

    # -------------------------------------------------------------
    # 16. Text Hybrid Recall vs. Computational Cost Pareto Trade-Off
    # -------------------------------------------------------------
    multi_path = os.path.join(RESULTS_DIR, 'recall_constrained_multi_method_comparison.csv')
    if os.path.exists(multi_path):
        df_multi = pd.read_csv(multi_path)
        df_text_m = df_multi[df_multi['Modality'] == 'Text'].copy()

        plt.figure(figsize=(9, 5.5), dpi=300)
        colors = {'Empirical Method': '#2b5c8f', 'Proposed Method': '#2ca02c', 'Baseline': '#7570b3', 'Scientific Control (Ablation)': '#d95f02'}

        for ctrl_type, grp in df_text_m.groupby('Control_Type'):
            plt.scatter(grp['Expensive_Call_Rate'] * 100.0, grp['Achieved_Candidate_Recall'] * 100.0,
                        s=130, color=colors.get(ctrl_type, '#333333'), label=ctrl_type, zorder=3)
            for _, r in grp.iterrows():
                short_name = r['Method'].split('(')[0].strip()
                plt.annotate(f"{short_name} ({r['Target_Candidate_Recall']})",
                             (r['Expensive_Call_Rate'] * 100.0, r['Achieved_Candidate_Recall'] * 100.0),
                             textcoords="offset points", xytext=(0, 7), ha='center', fontsize=7.5, fontweight='bold')

        plt.xlabel('Expensive Detector (SBERT) Call Rate (%)', fontsize=11)
        plt.ylabel('Candidate Recall (%)', fontsize=11)
        plt.title('Text Plagiarism Detection: Candidate Recall vs. SBERT Call Rate\n(Pareto Efficiency across 0.95 and 0.98 Constraints)', fontsize=13, fontweight='bold', pad=12)
        plt.legend(loc='lower right')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        p16 = os.path.join(FIG_DIR, '16_text_hybrid_recall_vs_cost.png')
        plt.savefig(p16, bbox_inches='tight')
        plt.close()
        print(f"Saved {p16}")

    # -------------------------------------------------------------
    # 17. Expensive Detector Call Rate and Routing Breakdown
    # -------------------------------------------------------------
    routing_path = os.path.join(RESULTS_DIR, 'setu_hybrid_routing_analysis.csv')
    if os.path.exists(routing_path):
        df_rout = pd.read_csv(routing_path)
        plt.figure(figsize=(7, 6), dpi=300)

        colors_pie = ['#386cb0', '#fdb462', '#2ca02c']
        labels = [f"{r['Routing_State']}\n({r['Percentage']:.1f}%)" for _, r in df_rout.iterrows()]

        wedges, texts = plt.pie(
            df_rout['Percentage'],
            labels=labels,
            colors=colors_pie,
            startangle=140,
            wedgeprops=dict(width=0.45, edgecolor='w', linewidth=2)
        )
        for t in texts:
            t.set_fontsize(10)
            t.set_fontweight('bold')

        plt.title('Setu E2 Staged Hybrid Routing Distribution\n(Workload Reduction via Selective Escalation)', fontsize=13, fontweight='bold', pad=12)
        plt.tight_layout()
        p17 = os.path.join(FIG_DIR, '17_text_detector_call_rate.png')
        plt.savefig(p17, bbox_inches='tight')
        plt.close()
        print(f"Saved {p17}")


if __name__ == '__main__':
    generate_all_visualizations()

