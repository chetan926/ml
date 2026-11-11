import os
import re
import pandas as pd
import numpy as np

TEXT_RAW_DIR = 'datasets/raw/text_mit_plagiarism'
CODE_RAW_DIR = 'datasets/raw/ir_plag/extracted/IR-Plag-Dataset'
PROCESSED_DIR = 'datasets/processed'

def prepare_text_dataset(sample_size=None, random_state=42):
    """
    Loads, cleans, and standardizes the MIT/SNLI Plagiarism Detection benchmark.
    Label mapping:
      label 0 (entailment/paraphrase) -> 1 (plagiarized)
      label 1 (neutral) -> 0 (non-plagiarized)
      label 2 (contradiction) -> 0 (non-plagiarized)
      label -1 -> dropped (unlabeled/ambiguous)
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    val_path = os.path.join(TEXT_RAW_DIR, 'snli_val.parquet')
    test_path = os.path.join(TEXT_RAW_DIR, 'snli_test.parquet')

    df_val = pd.read_parquet(val_path)
    df_test = pd.read_parquet(test_path)

    # Filter out unlabeled (-1)
    df_val = df_val[df_val['label'].isin([0, 1, 2])].copy()
    df_test = df_test[df_test['label'].isin([0, 1, 2])].copy()

    # Map to binary plagiarism label
    df_val['label_binary'] = (df_val['label'] == 0).astype(int)
    df_test['label_binary'] = (df_test['label'] == 0).astype(int)

    # Rename columns for domain clarity
    df_val = df_val.rename(columns={'premise': 'reference_text', 'hypothesis': 'suspicious_text', 'label_binary': 'is_plagiarized'})
    df_test = df_test.rename(columns={'premise': 'reference_text', 'hypothesis': 'suspicious_text', 'label_binary': 'is_plagiarized'})

    # Drop nulls or empty
    df_val = df_val.dropna(subset=['reference_text', 'suspicious_text']).reset_index(drop=True)
    df_test = df_test.dropna(subset=['reference_text', 'suspicious_text']).reset_index(drop=True)

    df_val['split'] = 'val'
    df_test['split'] = 'test'
    df_all = pd.concat([df_val, df_test], ignore_index=True)
    df_all['id'] = [f"TXT_{i:06d}" for i in range(len(df_all))]

    if sample_size and sample_size < len(df_all):
        df_all = df_all.sample(n=sample_size, random_state=random_state).reset_index(drop=True)

    output_path = os.path.join(PROCESSED_DIR, 'text_plagiarism_benchmark.csv')
    df_all[['id', 'reference_text', 'suspicious_text', 'label', 'is_plagiarized', 'split']].to_csv(output_path, index=False)
    print(f"[DataLoader] Saved cleaned text dataset to {output_path} ({len(df_all)} pairs)")
    return df_all

def prepare_code_dataset():
    """
    Parses Java files from the IR-Plag benchmark and generates paired comparison records.
    Cases: case-01 through case-07.
    Categories:
      - 'original' vs 'plagiarized' (L1-L6): is_plagiarized = 1
      - 'original' vs 'non-plagiarized': is_plagiarized = 0
      - 'non-plagiarized' vs 'non-plagiarized' (within same case): is_plagiarized = 0
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    if not os.path.exists(CODE_RAW_DIR):
        raise FileNotFoundError(f"IR-Plag raw directory not found at {CODE_RAW_DIR}")

    cases = sorted([d for d in os.listdir(CODE_RAW_DIR) if os.path.isdir(os.path.join(CODE_RAW_DIR, d))])
    
    pairs = []
    pair_id = 0

    for c in cases:
        c_path = os.path.join(CODE_RAW_DIR, c)
        
        # Load original code
        orig_dir = os.path.join(c_path, 'original')
        orig_files = [f for f in os.listdir(orig_dir) if f.endswith('.java')]
        if not orig_files:
            continue
        orig_file_path = os.path.join(orig_dir, orig_files[0])
        with open(orig_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            orig_code = f.read()

        # Load non-plagiarized submissions
        non_plag_dir = os.path.join(c_path, 'non-plagiarized')
        non_plag_submissions = []
        if os.path.exists(non_plag_dir):
            for sub in sorted(os.listdir(non_plag_dir)):
                sub_dir = os.path.join(non_plag_dir, sub)
                if os.path.isdir(sub_dir):
                    j_files = [f for f in os.listdir(sub_dir) if f.endswith('.java')]
                    if j_files:
                        with open(os.path.join(sub_dir, j_files[0]), 'r', encoding='utf-8', errors='ignore') as f:
                            code_content = f.read()
                        non_plag_submissions.append((sub, code_content))
                        
                        # Pair with original: label = 0 (independent work for same task)
                        pair_id += 1
                        pairs.append({
                            'id': f"CODE_{pair_id:05d}",
                            'case': c,
                            'reference_code': orig_code,
                            'suspicious_code': code_content,
                            'ref_type': 'original',
                            'sus_type': 'non-plagiarized',
                            'obfuscation_level': 'None',
                            'is_plagiarized': 0
                        })

        # Load plagiarized submissions across levels (L1 - L6)
        plag_dir = os.path.join(c_path, 'plagiarized')
        if os.path.exists(plag_dir):
            for level in sorted(os.listdir(plag_dir)):
                level_dir = os.path.join(plag_dir, level)
                if os.path.isdir(level_dir):
                    for sub in sorted(os.listdir(level_dir)):
                        sub_dir = os.path.join(level_dir, sub)
                        if os.path.isdir(sub_dir):
                            j_files = [f for f in os.listdir(sub_dir) if f.endswith('.java')]
                            if j_files:
                                with open(os.path.join(sub_dir, j_files[0]), 'r', encoding='utf-8', errors='ignore') as f:
                                    code_content = f.read()
                                
                                # Pair with original: label = 1 (plagiarized at level L1-L6)
                                pair_id += 1
                                pairs.append({
                                    'id': f"CODE_{pair_id:05d}",
                                    'case': c,
                                    'reference_code': orig_code,
                                    'suspicious_code': code_content,
                                    'ref_type': 'original',
                                    'sus_type': 'plagiarized',
                                    'obfuscation_level': level,
                                    'is_plagiarized': 1
                                })

        # Pair non-plagiarized against each other within task to represent cohort peer comparisons
        for i in range(len(non_plag_submissions)):
            for j in range(i + 1, min(i + 4, len(non_plag_submissions))):
                sub_a, code_a = non_plag_submissions[i]
                sub_b, code_b = non_plag_submissions[j]
                pair_id += 1
                pairs.append({
                    'id': f"CODE_{pair_id:05d}",
                    'case': c,
                    'reference_code': code_a,
                    'suspicious_code': code_b,
                    'ref_type': f"non-plag-{sub_a}",
                    'sus_type': f"non-plag-{sub_b}",
                    'obfuscation_level': 'None',
                    'is_plagiarized': 0
                })

    df_code = pd.DataFrame(pairs)
    output_path = os.path.join(PROCESSED_DIR, 'code_plagiarism_benchmark.csv')
    df_code.to_csv(output_path, index=False)
    print(f"[DataLoader] Saved cleaned source code dataset to {output_path} ({len(df_code)} pairs)")
    return df_code

def compute_dataset_statistics():
    """Generates comprehensive dataset statistics and summary tables."""
    text_path = os.path.join(PROCESSED_DIR, 'text_plagiarism_benchmark.csv')
    code_path = os.path.join(PROCESSED_DIR, 'code_plagiarism_benchmark.csv')

    stats = {}
    if os.path.exists(text_path):
        df_text = pd.read_csv(text_path)
        stats['text'] = {
            'total_pairs': len(df_text),
            'plagiarized_count': int(df_text['is_plagiarized'].sum()),
            'non_plagiarized_count': int((df_text['is_plagiarized'] == 0).sum()),
            'plagiarism_prevalence': float(df_text['is_plagiarized'].mean()),
            'avg_ref_len_words': float(df_text['reference_text'].apply(lambda x: len(str(x).split())).mean()),
            'avg_sus_len_words': float(df_text['suspicious_text'].apply(lambda x: len(str(x).split())).mean()),
            'val_count': int((df_text['split'] == 'val').sum()) if 'split' in df_text.columns else 0,
            'test_count': int((df_text['split'] == 'test').sum()) if 'split' in df_text.columns else 0,
        }

    if os.path.exists(code_path):
        df_code = pd.read_csv(code_path)
        stats['code'] = {
            'total_pairs': len(df_code),
            'plagiarized_count': int(df_code['is_plagiarized'].sum()),
            'non_plagiarized_count': int((df_code['is_plagiarized'] == 0).sum()),
            'plagiarism_prevalence': float(df_code['is_plagiarized'].mean()),
            'tasks_count': int(df_code['case'].nunique()),
            'obfuscation_levels': df_code['obfuscation_level'].value_counts().to_dict(),
            'avg_ref_lines': float(df_code['reference_code'].apply(lambda x: len(str(x).splitlines())).mean()),
            'avg_sus_lines': float(df_code['suspicious_code'].apply(lambda x: len(str(x).splitlines())).mean()),
        }

    print("\n================ DATASET AUDIT STATISTICS ================")
    if 'text' in stats:
        t = stats['text']
        print(f"TEXT BENCHMARK (MIT / SNLI Adaptation):")
        print(f"  Total Pairs: {t['total_pairs']}")
        print(f"  Plagiarized Pairs: {t['plagiarized_count']} ({t['plagiarism_prevalence']*100:.2f}%)")
        print(f"  Non-Plagiarized Pairs: {t['non_plagiarized_count']} ({(1-t['plagiarism_prevalence'])*100:.2f}%)")
        print(f"  Avg Ref Words: {t['avg_ref_len_words']:.1f} | Avg Sus Words: {t['avg_sus_len_words']:.1f}")
        print(f"  Validation Split: {t['val_count']} | Test Split: {t['test_count']}")

    if 'code' in stats:
        c = stats['code']
        print(f"\nSOURCE-CODE BENCHMARK (IR-Plag Java Corpus):")
        print(f"  Total Evaluated Pairs: {c['total_pairs']}")
        print(f"  Plagiarized Pairs: {c['plagiarized_count']} ({c['plagiarism_prevalence']*100:.2f}%)")
        print(f"  Non-Plagiarized Pairs: {c['non_plagiarized_count']} ({(1-c['plagiarism_prevalence'])*100:.2f}%)")
        print(f"  Introductory Assessment Tasks: {c['tasks_count']}")
        print(f"  Obfuscation Breakdown: {c['obfuscation_levels']}")
        print(f"  Avg Ref Lines: {c['avg_ref_lines']:.1f} | Avg Sus Lines: {c['avg_sus_lines']:.1f}")
    print("==========================================================\n")

    # Save statistics table
    rows = []
    if 'text' in stats:
        t = stats['text']
        rows.append({'Dataset': 'Text (MIT/SNLI Benchmark)', 'Modality': 'Natural Language Text', 'Total Pairs': t['total_pairs'], 'Plagiarized': t['plagiarized_count'], 'Non-Plagiarized': t['non_plagiarized_count'], 'Prevalence (%)': round(t['plagiarism_prevalence']*100, 2), 'Avg Size': f"{t['avg_ref_len_words']:.1f} words"})
    if 'code' in stats:
        c = stats['code']
        rows.append({'Dataset': 'Source Code (IR-Plag Java)', 'Modality': 'Java Source Code', 'Total Pairs': c['total_pairs'], 'Plagiarized': c['plagiarized_count'], 'Non-Plagiarized': c['non_plagiarized_count'], 'Prevalence (%)': round(c['plagiarism_prevalence']*100, 2), 'Avg Size': f"{c['avg_ref_lines']:.1f} lines"})
    df_summary = pd.DataFrame(rows)
    df_summary.to_csv('experiments/results/dataset_statistics_summary.csv', index=False)
    print("Saved experiments/results/dataset_statistics_summary.csv")
    return stats

if __name__ == '__main__':
    prepare_text_dataset()
    prepare_code_dataset()
    compute_dataset_statistics()

