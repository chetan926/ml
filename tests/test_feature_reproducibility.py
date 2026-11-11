"""
Reproducibility and Determinism Tests for Multi-View Features (Phase 3.2)
Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection
"""

import unittest
import numpy as np
import pandas as pd
from research_upgrade.src.features.feature_pipeline import (
    TextMultiViewPipeline,
    CodeMultiViewPipeline,
    TEXT_FEATURE_COLUMNS,
    CODE_FEATURE_COLUMNS
)
from research_upgrade.src.config import RANDOM_SEED


TEXT_CORPUS = pd.DataFrame([
    {
        "id": "repro_text_01",
        "reference_text": "Transformer architectures utilize multi-head self-attention mechanisms to capture syntactic dependencies.",
        "suspicious_text": "Attention mechanisms in multi-head transformers capture syntactic and semantic dependencies.",
        "is_plagiarized": 1,
        "split": "train",
        "premise_group_id": "group_rep_1"
    },
    {
        "id": "repro_text_02",
        "reference_text": "Algorithmic game theory analyzes strategic equilibrium in multi-agent economic settings.",
        "suspicious_text": "The cellular ribosome translates messenger RNA transcripts into functional polypeptide sequences.",
        "is_plagiarized": 0,
        "split": "train",
        "premise_group_id": "group_rep_2"
    },
    {
        "id": "repro_text_03",
        "reference_text": "Stochastic gradient descent updates model parameters proportionally to the negative gradient of the loss.",
        "suspicious_text": "Parameter vectors are updated along the negative objective gradient using stochastic optimization.",
        "is_plagiarized": 1,
        "split": "test",
        "premise_group_id": "group_rep_3"
    }
])

CODE_CORPUS = pd.DataFrame([
    {
        "id": "repro_code_01",
        "reference_code": "public class BinarySearch { public static int search(int[] a, int x) { int l=0, r=a.length-1; while(l<=r){ int m=(l+r)/2; if(a[m]==x) return m; if(a[m]<x) l=m+1; else r=m-1; } return -1; } }",
        "suspicious_code": "public class Search { public static int find(int[] arr, int target) { int low=0, high=arr.length-1; while(low<=high){ int mid=(low+high)/2; if(arr[mid]==target) return mid; if(arr[mid]<target) low=mid+1; else high=mid-1; } return -1; } }",
        "case": "case-01",
        "ref_type": "original",
        "sus_type": "plagiarized",
        "obfuscation_level": "L1",
        "is_plagiarized": 1
    },
    {
        "id": "repro_code_02",
        "reference_code": "public class HelloWorld { public static void main(String[] args) { System.out.println(\"Hello\"); } }",
        "suspicious_code": "public class Goodbye { public static void main(String[] args) { System.out.println(\"Bye\"); } }",
        "case": "case-02",
        "ref_type": "original",
        "sus_type": "non-plagiarized",
        "obfuscation_level": "NONE",
        "is_plagiarized": 0
    }
])


class TestFeatureReproducibility(unittest.TestCase):
    def test_text_feature_reproducibility(self):
        train_df = TEXT_CORPUS[TEXT_CORPUS["split"] == "train"]
        test_df = TEXT_CORPUS[TEXT_CORPUS["split"] == "test"]

        # Run 1
        pipe1 = TextMultiViewPipeline()
        pipe1.fit(train_df)
        df1 = pipe1.transform_dataframe(test_df, partition_name="test")

        # Run 2
        pipe2 = TextMultiViewPipeline()
        pipe2.fit(train_df)
        df2 = pipe2.transform_dataframe(test_df, partition_name="test")

        # Check identical values
        for col in TEXT_FEATURE_COLUMNS:
            vals1 = df1[col].to_numpy()
            vals2 = df2[col].to_numpy()
            np.testing.assert_allclose(
                vals1, vals2, atol=1e-5,
                err_msg=f"Feature {col} diverged across runs!"
            )

    def test_code_feature_reproducibility(self):
        pipe1 = CodeMultiViewPipeline()
        df1 = pipe1.transform_dataframe(CODE_CORPUS, policy="policy_a")

        pipe2 = CodeMultiViewPipeline()
        df2 = pipe2.transform_dataframe(CODE_CORPUS, policy="policy_a")

        for col in CODE_FEATURE_COLUMNS:
            vals1 = df1[col].to_numpy()
            vals2 = df2[col].to_numpy()
            np.testing.assert_allclose(
                vals1, vals2, atol=1e-5,
                err_msg=f"Code feature {col} diverged across runs!"
            )


if __name__ == "__main__":
    unittest.main()

