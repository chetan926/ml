import json
import os

def create_notebook(filename, cells):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3.11",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.18"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)
    print(f"Generated {filename}")

def build_all_notebooks():
    os.makedirs('notebooks', exist_ok=True)

    # 1. 01_dataset_analysis.ipynb
    cells_01 = [
        {"cell_type": "markdown", "metadata": {}, "source": [
            "# 01. Dataset Analysis and Preprocessing Audit\n",
            "### Research Project: Typed-Decision Pre-Filtering for Cost-Efficient Academic Integrity Detection Pipelines\n",
            "\n",
            "This notebook audits and analyzes the two primary benchmarks used in this research:\n",
            "1. **Text Plagiarism Benchmark**: Stanford Natural Language Inference (SNLI) Adaptation / MIT Plagiarism Corpus.\n",
            "2. **Source Code Plagiarism Benchmark**: IR-Plag Java Corpus (Oscar Karnalim, 2019).\n"
        ]},
        {"cell_type": "code", "execution_count": 1, "metadata": {}, "outputs": [], "source": [
            "import os\n",
            "import sys\n",
            "sys.path.insert(0, '../src')\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "\n",
            "# Load processed datasets\n",
            "df_text = pd.read_csv('../datasets/processed/text_plagiarism_benchmark.csv')\n",
            "df_code = pd.read_csv('../datasets/processed/code_plagiarism_benchmark.csv')\n",
            "\n",
            "print(f'Text Benchmark: {len(df_text)} pairs')\n",
            "print(f'Code Benchmark: {len(df_code)} pairs')\n"
        ]},
        {"cell_type": "code", "execution_count": 2, "metadata": {}, "outputs": [], "source": [
            "# Text Class Distribution\n",
            "print('Text Label Distribution:')\n",
            "print(df_text['is_plagiarized'].value_counts(normalize=True))\n",
            "\n",
            "# Code Obfuscation Distribution\n",
            "print('\\nCode Obfuscation Levels in IR-Plag:')\n",
            "print(df_code['obfuscation_level'].value_counts())\n"
        ]}
    ]
    create_notebook('notebooks/01_dataset_analysis.ipynb', cells_01)

    # 2. 02_text_methods.ipynb
    cells_02 = [
        {"cell_type": "markdown", "metadata": {}, "source": [
            "# 02. Text Plagiarism Detection Methodologies\n",
            "### Comparing Method 1 (TF-IDF Lexical Baseline) & Method 2 (Contextual SBERT Semantic Detector)\n",
            "\n",
            "- **Method 1**: TF-IDF + Cosine Similarity (Sajid et al., 2025, *PLOS ONE*)\n",
            "- **Method 2**: Contextual Sentence-BERT Embeddings (Setu et al., 2025, *Springer*)\n"
        ]},
        {"cell_type": "code", "execution_count": 1, "metadata": {}, "outputs": [], "source": [
            "import sys\n",
            "sys.path.insert(0, '../src')\n",
            "import pandas as pd\n",
            "from text_tfidf import TextTFIDFDetector\n",
            "from bert_similarity import ContextualSemanticDetector\n",
            "\n",
            "df_text_summary = pd.read_csv('../experiments/results/text_experiments_summary.csv')\n",
            "df_text_summary[['Method / Condition', 'Accuracy', 'Precision', 'Recall', 'F1', 'Avg Latency (ms)']]\n"
        ]}
    ]
    create_notebook('notebooks/02_text_methods.ipynb', cells_02)

    # 3. 03_code_methods.ipynb
    cells_03 = [
        {"cell_type": "markdown", "metadata": {}, "source": [
            "# 03. Source Code Plagiarism Detection Methodologies\n",
            "### Method 3: Syntactic & Token SVM Plagiarism Detector (Gandhi et al., 2024 adapted to Java)\n",
            "\n",
            "Adapts Gandhi et al. (2024, *Frontiers in Computer Science*) from Python to Java on the IR-Plag benchmark.\n"
        ]},
        {"cell_type": "code", "execution_count": 1, "metadata": {}, "outputs": [], "source": [
            "import sys\n",
            "sys.path.insert(0, '../src')\n",
            "import pandas as pd\n",
            "df_code_summary = pd.read_csv('../experiments/results/code_experiments_summary.csv')\n",
            "df_obf = pd.read_csv('../experiments/results/code_obfuscation_level_recall.csv')\n",
            "\n",
            "display(df_code_summary[['Method / Condition', 'Accuracy', 'Precision', 'Recall', 'F1', 'Avg Latency (ms)']])\n",
            "display(df_obf)\n"
        ]}
    ]
    create_notebook('notebooks/03_code_methods.ipynb', cells_03)

    # 4. 04_typed_decision_pipeline.ipynb
    cells_04 = [
        {"cell_type": "markdown", "metadata": {}, "source": [
            "# 04. Proposed Typed-Decision Pre-Filtering Pipeline\n",
            "### Architectural Implementation, Routing & Tri-Band Confidence Escalation\n",
            "\n",
            "Demonstrates the core research contribution:\n",
            "- C1: Typed Routing (Text vs Java Code)\n",
            "- C2: Tri-Band Confidence / Risk Filtering\n",
            "- C3: Cost-Aware Escalation on Uncertain Band\n",
            "- C4: Validation-Tuned Threshold Boundaries\n"
        ]},
        {"cell_type": "code", "execution_count": 1, "metadata": {}, "outputs": [], "source": [
            "import sys\n",
            "sys.path.insert(0, '../src')\n",
            "from typed_decision_pipeline import TypedDecisionPipeline\n",
            "\n",
            "pipeline = TypedDecisionPipeline()\n",
            "print('Pipeline Modality Detection Test:')\n",
            "sample_code = 'public class Test { public static void main(String[] args) {} }'\n",
            "sample_text = 'Academic integrity requires students to submit their own authentic work.'\n",
            "\n",
            "print('Sample 1 Routing:', pipeline.detect_modality(sample_code, sample_code))\n",
            "print('Sample 2 Routing:', pipeline.detect_modality(sample_text, sample_text))\n"
        ]}
    ]
    create_notebook('notebooks/04_typed_decision_pipeline.ipynb', cells_04)

    # 5. 05_comparative_analysis.ipynb
    cells_05 = [
        {"cell_type": "markdown", "metadata": {}, "source": [
            "# 05. Comparative Evaluation & Experimental Visualizations\n",
            "### Operating Conditions Comparison: Condition A (Full) vs B (Fixed) vs C (Proposed Pipeline)\n"
        ]},
        {"cell_type": "code", "execution_count": 1, "metadata": {}, "outputs": [], "source": [
            "import os\n",
            "from IPython.display import Image, display\n",
            "\n",
            "figures = sorted(os.listdir('../experiments/figures'))\n",
            "for fig in figures:\n",
            "    if fig.endswith('.png'):\n",
            "        print(f'Displaying: {fig}')\n",
            "        display(Image(filename=os.path.join('../experiments/figures', fig)))\n"
        ]}
    ]
    create_notebook('notebooks/05_comparative_analysis.ipynb', cells_05)

if __name__ == '__main__':
    build_all_notebooks()

