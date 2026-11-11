import os
import pandas as pd

def inspect():
    print("--- 1. Inspecting Text Dataset (MIT / SNLI Benchmark) ---")
    val_path = 'datasets/raw/text_mit_plagiarism/snli_val.parquet'
    test_path = 'datasets/raw/text_mit_plagiarism/snli_test.parquet'
    
    if os.path.exists(val_path):
        df_val = pd.read_parquet(val_path)
        print(f"Validation Parquet: {df_val.shape[0]} rows, columns: {df_val.columns.tolist()}")
        print("Validation label distribution:\n", df_val['label'].value_counts())
        print("Sample pair:")
        sample = df_val.iloc[0]
        print(f"Premise: {sample['premise']}")
        print(f"Hypothesis: {sample['hypothesis']}")
        print(f"Label: {sample['label']}")
        
    if os.path.exists(test_path):
        df_test = pd.read_parquet(test_path)
        print(f"\nTest Parquet: {df_test.shape[0]} rows, columns: {df_test.columns.tolist()}")
        print("Test label distribution:\n", df_test['label'].value_counts())

    print("\n--- 2. Inspecting Code Dataset (IR-Plag Java Benchmark) ---")
    ir_plag_dir = 'datasets/raw/ir_plag/extracted/IR-Plag-Dataset'
    if os.path.exists(ir_plag_dir):
        cases = sorted([d for d in os.listdir(ir_plag_dir) if os.path.isdir(os.path.join(ir_plag_dir, d))])
        print(f"Cases found ({len(cases)}): {cases}")
        for c in cases:
            c_dir = os.path.join(ir_plag_dir, c)
            subs = os.listdir(c_dir)
            orig_files = len(os.listdir(os.path.join(c_dir, 'original'))) if 'original' in subs else 0
            non_plag = len(os.listdir(os.path.join(c_dir, 'non-plagiarized'))) if 'non-plagiarized' in subs else 0
            plag = len(os.listdir(os.path.join(c_dir, 'plagiarized'))) if 'plagiarized' in subs else 0
            print(f"  {c}: original={orig_files}, non-plagiarized={non_plag}, plagiarized_levels={plag}")

if __name__ == '__main__':
    inspect()

