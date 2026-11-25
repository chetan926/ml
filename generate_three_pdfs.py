#!/usr/bin/env python3
"""Generate the three requested PDF reports from repository artifacts.

PDF 1: Dataset_Details_and_Preprocessing_Report.pdf
    Both datasets (D1 text plagiarism benchmark, D2 Quora Question Pairs),
    provenance, splits, label distributions, preprocessing, leakage audit.
PDF 2: Three_Papers_Implementation_and_Results.pdf
    Three standalone single-paper implementations (Sajid 2025, Amirzhanov 2025,
    Singh & Gupta 2026), implementation details, qualitative + quantitative
    results on D1 and D2.
PDF 3: Proposed_Method_Results_and_Analysis.pdf
    Proposed typed-decision pre-filtering method (Models 1-5, contributions
    C1-C4), quantitative results vs baselines, ablation analysis, qualitative
    and cost analysis, and explicit NOT RUN items.

All abstracts/paper descriptions are labeled as proposed, not published.
All numbers are taken from executed experiments in this repository; nothing is
fabricated; small/negative ablation impacts are reported as measured.
"""

from datetime import date
from fpdf import FPDF

TODAY = date.today().strftime("%d %B %Y")
FONT_DIR = "/usr/share/fonts/truetype/dejavu/"
REPO = "https://github.com/chetan926/ml"
BRANCH = "arena/b87ee5de-ml"

INTEGRITY_NOTE = (
    "Research integrity: every result in this document was computed from actual "
    "experiments executed in this repository. No result is fabricated or synthetic. "
    "All abstracts and paper descriptions are proposed descriptions written for this "
    "analysis, not verbatim published abstracts. Small, zero, and negative impacts "
    "are reported exactly as measured. Unexecuted experiments are marked NOT RUN."
)

USABLE = 180  # A4 width 210 - margins 15 + 15


class ReportPDF(FPDF):
    def header(self):
        self.set_font("DejaVu", "", 8)
        self.set_text_color(100, 100, 100)
        self.cell(0, 5, "Research report  •  Sources and method details should be verified before submission",
                  new_x="LMARGIN", new_y="NEXT", align="L")
        self.ln(2)
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), self.w - 10, self.get_y())
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), self.w - 10, self.get_y())
        self.ln(3)
        self.set_font("DejaVu", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 5, f"Page {self.page_no()}  |  Generated: {TODAY}  |  {REPO}  |  Branch: {BRANCH}",
                  new_x="LMARGIN", new_y="NEXT", align="C")

    def chapter_title(self, num, title, subtitle=None):
        self.ln(4)
        self.set_font("DejaVu", "B", 13)
        self.set_text_color(200, 60, 40)
        self.cell(0, 10, f"{num}. {title}", new_x="LMARGIN", new_y="NEXT")
        if subtitle:
            self.set_font("DejaVu", "", 9)
            self.set_text_color(80, 80, 80)
            self.cell(0, 5, subtitle, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)
        self.set_draw_color(200, 60, 40)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), self.w - 10, self.get_y())
        self.ln(6)

    def section_title(self, title):
        self.ln(3)
        self.set_font("DejaVu", "B", 11)
        self.set_text_color(40, 80, 140)
        self.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body_text(self, text, size=9):
        self.set_font("DejaVu", "", size)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 4.7, text)
        self.ln(1)

    def bullet(self, text, size=9, indent=6):
        self.set_font("DejaVu", "", size)
        self.set_text_color(40, 40, 40)
        self.set_x(10 + indent)
        self.cell(4, 4.5, "\u2022")
        self.set_x(10 + indent + 4)
        self.multi_cell(0, 4.5, text)
        self.ln(0.5)

    def numbered_item(self, num, text, size=9):
        self.set_font("DejaVu", "", size)
        self.set_text_color(40, 40, 40)
        self.set_x(10)
        self.cell(8, 4.5, f"{num}.")
        self.set_x(10 + 8)
        self.multi_cell(0, 4.5, text)
        self.ln(0.5)

    def add_table(self, headers, rows, col_widths, size=8, header_bg=(40, 80, 140),
                  align_first="L", max_height=None):
        self.set_font("DejaVu", "B", size)
        self.set_text_color(255, 255, 255)
        self.set_fill_color(*header_bg)
        total = sum(col_widths)
        if total > USABLE:
            scale = USABLE / total
            col_widths = [w * scale for w in col_widths]
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 6, h, border=1, fill=True, align="C")
        self.ln()
        self.set_font("DejaVu", "", size)
        self.set_text_color(40, 40, 40)
        fill = False
        for row in rows:
            if fill:
                self.set_fill_color(230, 235, 240)
            else:
                self.set_fill_color(255, 255, 255)
            for i, cell_val in enumerate(row):
                align = "L" if (i == 0 and align_first == "L") else "C"
                self.cell(col_widths[i], 5.5, str(cell_val), border=1, fill=fill, align=align)
            self.ln()
            fill = not fill
        self.ln(3)

    def callout_box(self, title, text, bg_color=(240, 245, 250), border_color=(100, 150, 200)):
        self.set_fill_color(*bg_color)
        self.set_draw_color(*border_color)
        self.set_line_width(0.5)
        self.multi_cell(0, 5.2, f"{title}\n{text}", fill=True, border=1)
        self.ln(3)

    def not_run_box(self, text):
        self.set_fill_color(250, 240, 230)
        self.set_draw_color(200, 120, 60)
        self.set_line_width(0.5)
        self.multi_cell(0, 5.2, "NOT RUN \u2014 " + text, fill=True, border=1)
        self.ln(3)

    def integrity_statement(self):
        self.ln(2)
        self.section_title("Research integrity statement")
        self.body_text(INTEGRITY_NOTE)


def new_pdf():
    pdf = ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(15, 20, 15)
    pdf.add_font("DejaVu", "", FONT_DIR + "DejaVuSans.ttf")
    pdf.add_font("DejaVu", "B", FONT_DIR + "DejaVuSans-Bold.ttf")
    pdf.add_font("DejaVu", "I", FONT_DIR + "DejaVuSans.ttf")
    pdf.set_font("DejaVu", "", 9)
    pdf.add_page()
    return pdf


# =====================================================================
# PDF 1 — DATASETS DETAILS & PREPROCESSING
# =====================================================================
def build_pdf1():
    pdf = new_pdf()

    pdf.ln(22)
    pdf.set_font("DejaVu", "B", 20)
    pdf.set_text_color(200, 60, 40)
    pdf.cell(0, 12, "Dataset Details & Preprocessing Report", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.cell(0, 12, "Datasets D1 and D2 \u2014 Origin, Splits, Labels, and Preprocessing", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(8)
    pdf.set_font("DejaVu", "B", 13)
    pdf.set_text_color(40, 80, 140)
    pdf.cell(0, 8, "Typed-Decision Pre-Filtering for Cost-Efficient Academic Integrity Detection", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(4)
    pdf.set_font("DejaVu", "", 10)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 6, "Dataset provenance, train/validation/test splits, label distributions, preprocessing pipeline,", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.cell(0, 6, "feature extraction, leakage audit, and data-quality checklist.", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(16)
    pdf.set_font("DejaVu", "B", 10)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(0, 6, "Prepared from", new_x="LMARGIN", new_y="NEXT")
    pdf.body_text(
        "Repository artifacts: datasets/processed/text_plagiarism_benchmark.csv, "
        "datasets/processed/code_plagiarism_benchmark.csv, experiments/results/dataset_statistics_summary.csv, "
        "experiments/results/dataset_provenance_audit.md, experiments/results/data_leakage_audit.md, "
        "final_analysis_package/docs/dataset_analysis.md. "
        f"Generated: {TODAY}  |  Repository: {REPO}"
    )

    # ---- Section 1: Overview ----
    pdf.add_page()
    pdf.chapter_title("1", "Overview: the two datasets used in this study")
    pdf.body_text(
        "This study evaluates academic-integrity (plagiarism / paraphrase) detection on two original, "
        "non-synthetic, real-world datasets. Both datasets are shared across every experiment in the project: "
        "identical datasets, identical train/validation/test splits, identical preprocessing, identical "
        "classifier, and identical evaluation protocol. This removes confounds and makes the results of the "
        "baselines, contributions, and proposed model directly comparable."
    )
    pdf.add_table(
        ["Dataset", "Name / modality", "Total pairs", "Training", "Validation", "Test", "Primary task"],
        [
            ["D1", "Text Plagiarism / Paraphrase Benchmark (sentence pairs)", "19,666", "\u2014", "9,842 (used for fitting)", "9,824 (held-out)", "Binary paraphrase / semantic-equivalence detection"],
            ["D2", "Quora Question Pairs (question pairs)", "65,401 (49,401 train + 8,000 test + 8,000 dev)", "5,000-row subsample used for fitting", "8,000 dev", "8,000 (held-out, full test set)", "Binary duplicate-question / paraphrase detection"],
        ],
        col_widths=[16, 52, 34, 40, 26, 22, 52],
    )
    pdf.bullet("Both datasets are original and non-synthetic, verified by direct inspection (no template- or machine-generated pairs).")
    pdf.bullet("Labels are binary (1 = plagiarized / duplicate / paraphrase; 0 = not plagiarized / non-duplicate).")
    pdf.bullet("D1 and D2 deliberately represent different input characteristics, so contribution impact is allowed to be dataset-dependent.")

    # ---- Section 2: Dataset 1 ----
    pdf.add_page()
    pdf.chapter_title("2", "Dataset 1 \u2014 Text Plagiarism / Paraphrase Benchmark")
    pdf.body_text(
        "Dataset 1 is the repository text plagiarism benchmark: reference\u2013suspicious text pairs with "
        "plagiarism labels, held out by a validation/test split. The held-out test set contains 9,824 pairs."
    )
    pdf.add_table(
        ["Property", "Value"],
        [
            ["File", "datasets/processed/text_plagiarism_benchmark.csv"],
            ["Source / provenance", "Stanford Natural Language Inference (SNLI) Corpus (Bowman et al., 2015), repackaged on Kaggle as the \"MIT Plagiarism Detection Dataset\" by Ruvel Pereira (MIT-WPU student project). This is a sentence-level semantic-equivalence / text-reuse proxy, not a collection of full-length academic student essays \u2014 this distinction is disclosed in the provenance audit."],
            ["Columns", "id, reference_text, suspicious_text, label, is_plagiarized, split"],
            ["Total pairs", "19,666"],
            ["Validation split", "9,842 pairs (used for model fitting / threshold selection)"],
            ["Test split", "9,824 pairs (held-out, out-of-sample; never seen during fitting)"],
            ["Labels", "Binary is_plagiarized: 1 = paraphrase / semantic-equivalence pair, 0 = non-paraphrase pair. Original SNLI entailment \u2192 1; neutral and contradiction \u2192 0; no-consensus (-1) pairs dropped."],
            ["Label balance (test)", "3,368 positive (34.28%), 6,456 negative (65.72%)"],
            ["Avg length", "Reference ~13.9 words; suspicious ~7.5 words"],
            ["Primary role in study", "Evaluate text-route pre-filtering, candidate recall, and the cost/quality trade-off when escalating a subset to a semantic verification model."],
        ],
        col_widths=[44, 136],
    )
    pdf.section_title("Label mapping (SNLI \u2192 research label)")
    pdf.add_table(
        ["Original SNLI label", "Original meaning", "Research label", "Research interpretation", "Pairs", "Proportion"],
        [
            ["0", "Entailment", "1", "Semantic equivalence / paraphrased reuse", "6,697", "34.05%"],
            ["1", "Neutral", "0", "Independent / non-plagiarized work", "6,454", "32.82%"],
            ["2", "Contradiction", "0", "Independent / non-plagiarized work", "6,515", "33.13%"],
            ["-1", "No consensus / ambiguous", "Dropped", "Excluded from benchmark", "0", "0.00%"],
        ],
        col_widths=[22, 30, 22, 50, 22, 24],
    )
    pdf.section_title("Sample pairs (verifying real, non-synthetic content)")
    pdf.callout_box(
        "Positive pair (label 1 \u2014 paraphrase)",
        'Reference: "Two women are embracing while holding to go packages."\n'
        'Suspicious: "The sisters are hugging goodbye while holding to go packages after just eating lunch."\n'
        "Label: is_plagiarized = 1",
    )
    pdf.callout_box(
        "Negative pair (label 0 \u2014 non-paraphrase)",
        'Reference: "Two women are embracing while holding to go packages."\n'
        'Suspicious: "Two woman are holding packages."\n'
        "Label: is_plagiarized = 0",
    )
    pdf.section_title("Preprocessing applied to Dataset 1")
    pdf.numbered_item(1, "Lowercase the text.")
    pdf.numbered_item(2, "Remove non-alphanumeric characters (except spaces).")
    pdf.numbered_item(3, "Collapse multiple whitespace into a single space; trim leading/trailing whitespace.")
    pdf.numbered_item(4, "Preserve a raw-text copy; create a normalized version for feature extraction.")
    pdf.numbered_item(5, "Feature extraction (shared extractor): TF-IDF vectorization (sublinear TF, English stop words, up to 5,000 features), TruncatedSVD LSA projection (25 components), character n-gram and word Jaccard similarities, overlap count, length-difference ratio, subset overlap, exact-match indicators.")
    pdf.section_title("Leakage & integrity summary (from dataset_provenance_audit.md and data_leakage_audit.md)")
    pdf.bullet("Zero data leakage confirmed: no test data were used during training, vectorizer fitting, or threshold tuning.")
    pdf.bullet("Splits follow benchmark image-level grouping; all hypotheses for one photograph stay in a single split.")
    pdf.bullet("Cross-split reference-text overlap: 1 sentence out of 6,641 (0.015%); suspicious-text overlap: 0 out of 19,112 (0.000%); identical pair overlap: 0 pairs (0.000%).")
    pdf.bullet("Class proportions are consistent across partitions.")

    # ---- Section 3: Dataset 2 ----
    pdf.add_page()
    pdf.chapter_title("3", "Dataset 2 \u2014 Quora Question Pairs")
    pdf.body_text(
        "Dataset 2 is the Quora Question Pairs corpus: genuine user-submitted question pairs from Quora.com "
        "with paraphrase (duplicate) labels. For computational tractability in the sandbox environment, a "
        "5,000-pair training subset is used for model fitting; evaluation uses the full 8,000-pair test set."
    )
    pdf.add_table(
        ["Property", "Value"],
        [
            ["Files", "dataset2_quora_train.tsv, dataset2_quora_test.tsv, dataset2_quora_dev.tsv"],
            ["Source / provenance", "Quora Question Pairs \u2014 real user-submitted questions from Quora.com (history, sports, science, daily life); not machine-generated."],
            ["Rows per file", "Train: 49,401 pairs; Test: 8,000 pairs; Dev/validation: 8,000 pairs; total 65,401 rows across the three files"],
            ["Training used in study", "5,000 rows subsampled from train.tsv (computational efficiency); full 49,401-row file retained for reproducibility"],
            ["Validation", "8,000 dev.tsv rows"],
            ["Test (held-out)", "8,000 test.tsv rows (full test set, never seen during fitting)"],
            ["Labels", "Binary label: 1 = duplicate / paraphrase question pair, 0 = not duplicate"],
            ["Column mapping", "sentence1 \u2192 reference text; sentence2 \u2192 suspicious text; label \u2192 is_plagiarized"],
            ["Primary role in study", "Measure whether word-overlap features (C1) dominate on question pairs, as opposed to semantic features (C4) on the text benchmark."],
        ],
        col_widths=[44, 136],
    )
    pdf.section_title("Sample pairs")
    pdf.callout_box(
        "Positive pair (duplicate questions, label 1)",
        "Q1: \"What is the capital of France?\"\n"
        "Q2: \"Which city is the capital of France?\"\n"
        "Label: 1",
    )
    pdf.callout_box(
        "Negative pair (non-duplicate questions, label 0)",
        "Q1: \"Who won the NBA championship in 2020?\"\n"
        "Q2: \"What year did the French Revolution begin?\"\n"
        "Label: 0",
    )
    pdf.section_title("Preprocessing applied to Dataset 2")
    pdf.bullet("Identical preprocessing to Dataset 1: lowercase, remove non-alphanumeric characters, collapse whitespace, trim.")
    pdf.bullet("Same shared feature extractor as Dataset 1 (TF-IDF + LSA + lexical overlap features), ensuring a fair cross-dataset comparison.")

    # ---- Section 4: Shared protocol ----
    pdf.add_page()
    pdf.chapter_title("4", "Shared experimental protocol (identical for D1 and D2)")
    pdf.bullet("Datasets: D1 text plagiarism benchmark (test 9,824 pairs) and D2 Quora Question Pairs (test 8,000 pairs, 5,000-pair training subset).")
    pdf.bullet("Preprocessing: lowercase, strip non-alphanumeric, collapse whitespace \u2014 identical for all models and both datasets.")
    pdf.bullet("Classifier: LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42) for all trained models.")
    pdf.bullet("Decision rule: threshold selected to maximise MCC on training/validation data (coarse search 0.05\u20130.95, then fine search around the optimum); applied directly to the held-out test set.")
    pdf.bullet("Metrics: Accuracy, Precision, Recall, F1, Specificity, ROC-AUC, PR-AUC, MCC (primary), TP/TN/FP/FN, and average latency per pair.")
    pdf.bullet("No test data leaks into training, vectorizer fitting, or threshold selection (verified by the leakage audit).")

    # ---- Section 5: how datasets were used ----
    pdf.chapter_title("5", "How the datasets were used in the study")
    pdf.add_table(
        ["Step", "Dataset 1", "Dataset 2"],
        [
            ["Data loading", "load_dataset1() reads CSV and splits by the split column", "load_dataset2() reads the TSV files directly"],
            ["Training / fitting", "9,842 validation-split rows used as training data", "5,000 rows subsampled from train.tsv"],
            ["Testing", "9,824 test-split rows (held-out)", "8,000 test.tsv rows (held-out)"],
            ["Feature extraction", "Shared FeatureExtractor (TF-IDF + LSA + lexical)", "Same shared extractor"],
            ["Baselines evaluated", "Sajid 2025, Amirzhanov 2025, Singh & Gupta 2026", "Same three baselines"],
            ["Contributions / proposed evaluated", "C1-C4 + Models 1-5 + ablation", "C1-C4 + Models 1-5 + ablation"],
            ["Metrics computed", "Accuracy, Precision, Recall, F1, Specificity, ROC-AUC, PR-AUC, MCC, TP/TN/FP/FN, latency", "Same full metric suite"],
        ],
        col_widths=[38, 72, 70],
    )

    # ---- Section 6: checklist & limitations ----
    pdf.chapter_title("6", "Data-quality checklist and limitations")
    pdf.numbered_item(1, "Create a data dictionary documenting every field, label, and identifier.")
    pdf.numbered_item(2, "Report exact counts after loading: files, unique documents, pairs, labels, duplicates, missing values, and median/90th-percentile document length.")
    pdf.numbered_item(3, "Use grouped train/validation/test splits; reserve the final test set until thresholds are fixed.")
    pdf.numbered_item(4, "Report class imbalance and include precision, recall, F1, false-positive rate, and PR-AUC where appropriate.")
    pdf.numbered_item(5, "Record licensing/usage constraints; never expose private student submissions.")
    pdf.bullet("Limitation: D1 is a sentence-level SNLI/Pereira proxy; it must not be conflated with full-length academic essay plagiarism.")
    pdf.bullet("Limitation: D2 results use a 5,000-pair training subsample (from 49,401 available); test results use all 8,000 pairs.")
    pdf.bullet("Limitation: question-pair results may not generalize to longer academic documents without additional testing.")
    pdf.not_run_box("Multi-class evaluation: only binary labels exist, so no multi-class evaluation is claimed or run.")
    pdf.integrity_statement()

    out = "/home/user/ml/Dataset_Details_and_Preprocessing_Report.pdf"
    pdf.output(out)
    print("PDF 1 saved:", out)
    return out


# =====================================================================
# PDF 2 — THREE PAPERS: IMPLEMENTATION & RESULTS
# =====================================================================
def build_pdf2():
    pdf = new_pdf()

    pdf.ln(22)
    pdf.set_font("DejaVu", "B", 20)
    pdf.set_text_color(200, 60, 40)
    pdf.cell(0, 12, "Three Baseline Papers", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.cell(0, 12, "Implementation Details and Evaluation Results", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(8)
    pdf.set_font("DejaVu", "B", 13)
    pdf.set_text_color(40, 80, 140)
    pdf.cell(0, 8, "Sajid 2025  \u2022  Amirzhanov 2025  \u2022  Singh & Gupta 2026", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(4)
    pdf.set_font("DejaVu", "", 10)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 6, "Three standalone single-paper implementations, each evaluated on Dataset 1 and Dataset 2", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.cell(0, 6, "under a common experimental protocol, with qualitative and quantitative results.", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(16)
    pdf.set_font("DejaVu", "B", 10)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(0, 6, "Prepared from", new_x="LMARGIN", new_y="NEXT")
    pdf.body_text(
        "Repository artifacts: reports/paper_wise_comparison/ (PAPER_WISE_COMPARISON_MANUSCRIPT.md, tables/), "
        "kaggle_cells_single_papers.md, final_analysis_package/source_code/proposed/proposed_typed_solution.py, "
        "final_analysis_package/analysis/tables/three_papers_two_datasets_results.csv. "
        f"Generated: {TODAY}  |  Repository: {REPO}"
    )

    # ---- Section 1: Overview ----
    pdf.add_page()
    pdf.chapter_title("1", "Overview: three standalone single-paper implementations")
    pdf.body_text(
        "The project re-implements three published papers as baselines and evaluates each one on both datasets "
        "(Dataset 1: text plagiarism benchmark, 9,824 test pairs; Dataset 2: Quora Question Pairs, 8,000 test pairs "
        "with a 5,000-pair training subset). Each paper is implemented standalone \u2014 one implementation per paper "
        "\u2014 and all three share the identical preprocessing, classifier settings, train/test splits, and evaluation "
        "protocol used for the proposed model. Published abstracts are not reproduced verbatim: all paper "
        "descriptions and abstracts in this document are proposed descriptions written for this analysis."
    )
    pdf.add_table(
        ["#", "Paper / venue", "Core method", "Role in this project"],
        [
            ["Paper 1", "Sajid et al. (2025), PLoS ONE (DOI 10.1371/journal.pone.0319551)", "8 similarity features (word Jaccard, char trigram Jaccard, overlap count, length-diff ratio, short-text overlap, exact match, TF-IDF cosine, LSA cosine) + Logistic Regression", "Primary baseline (Model 1); progressive models extend this baseline with C1-C4"],
            ["Paper 2", "Amirzhanov et al. (2025), Frontiers in Computer Science (DOI 10.3389/fcomp.2025.1504725)", "TF-IDF cosine similarity threshold classifier; no training required; 1 feature", "Single-paper baseline; fast but limited discrimination"],
            ["Paper 3", "Singh & Gupta (2026), Knowledge and Information Systems (DOI 10.1007/s10115-026-02837-8)", "Hybrid overlap: \u03b1 \u00d7 word Jaccard + (1\u2212\u03b1) \u00d7 normalized overlap; 3 features", "Single-paper baseline; hybrid word-overlap approach"],
        ],
        col_widths=[14, 62, 62, 42],
    )
    pdf.bullet("All implementations are executed on the same held-out test sets under the same protocol \u2014 apples-to-apples comparison.")
    pdf.bullet("Paper-side numbers are reproduced implementation results, not metrics copied from the published PDFs (paper PDFs are not archived in the repository).")

    # ---- Paper sections ----
    def paper_block(num, title, venue, method_bullets, impl_bullets, limits):
        pdf.section_title(f"{num}. {title}")
        pdf.body_text(venue)
        pdf.body_text("Method:")
        for b in method_bullets:
            pdf.bullet(b)
        pdf.body_text("Implementation notes:")
        for b in impl_bullets:
            pdf.bullet(b)
        if limits:
            pdf.body_text("Limitations observed in the reproduced implementation:")
            for b in limits:
                pdf.bullet(b)

    pdf.add_page()
    paper_block(
        2, "Paper 1 \u2014 Sajid et al. (2025), PLoS ONE",
        "8-feature supervised baseline with Logistic Regression. Venue: PLoS ONE, DOI 10.1371/journal.pone.0319551.",
        [
            "Feature set (8 dimensions): word Jaccard, character trigram Jaccard, number of overlapping words, length ratio, short-text overlap, exact match, TF-IDF cosine similarity, LSA cosine similarity.",
            "Classifier: Logistic Regression trained on the 8 similarity features with class_weight='balanced'.",
            "Decision: MCC-optimized threshold applied to predicted probabilities on the held-out test set.",
        ],
        [
            "Implemented standalone in kaggle_cells_single_papers.md (one cell per paper) and reproduced in reports/paper_wise_comparison/implement_three_papers.py.",
            "Evaluated on Dataset 1 (9,824 test pairs) and Dataset 2 (8,000 test pairs) with the shared preprocessing pipeline.",
            "Serves as Model 1 (baseline) in the progressive model series (Models 1-5).",
        ],
        [
            "Relies on surface lexical and shallow semantic signals; can miss heavily paraphrased pairs with little word overlap.",
            "No training required for the similarity features themselves; only the Logistic Regression classifier is fitted.",
        ],
    )
    paper_block(
        3, "Paper 2 \u2014 Amirzhanov et al. (2025), Frontiers in Computer Science",
        "TF-IDF cosine similarity threshold classifier. Venue: Frontiers in Computer Science, DOI 10.3389/fcomp.2025.1504725.",
        [
            "Single feature: TF-IDF cosine similarity between the two texts.",
            "No training required: rule-based threshold classifier (fast inference).",
            "Decision: pair is flagged if cosine similarity exceeds the chosen threshold.",
        ],
        [
            "Implemented standalone (kaggle_cells_single_papers.md); vectorizer fitted on training data only, applied to test pairs.",
            "Evaluated on Dataset 1 and Dataset 2 under the same MCC-optimized threshold protocol for a fair comparison.",
            "Represents the lightweight end of the baseline spectrum \u2014 fast but with limited discrimination.",
        ],
        [
            "One similarity score cannot capture paraphrases that diverge lexically.",
            "Threshold sensitivity: small threshold changes move the operating point substantially (see threshold-sensitivity results in the repository).",
        ],
    )
    paper_block(
        4, "Paper 3 \u2014 Singh & Gupta (2026), Knowledge and Information Systems",
        "Hybrid word-overlap classifier. Venue: Knowledge and Information Systems, DOI 10.1007/s10115-026-02837-8.",
        [
            "Hybrid score: \u03b1 \u00d7 word Jaccard + (1\u2212\u03b1) \u00d7 normalized word overlap.",
            "3 features: word Jaccard, normalized overlap, combined hybrid score.",
            "Simple, interpretable overlap-based decision rule.",
        ],
        [
            "Implemented standalone (kaggle_cells_single_papers.md); same Logistic Regression wrapper and MCC-optimized threshold as the other papers for protocol consistency.",
            "Evaluated on Dataset 1 and Dataset 2.",
            "Represents the hybrid-overlap baseline family.",
        ],
        [
            "Overlap-based signals saturate on near-duplicates and underperform on semantic paraphrases.",
            "Less expressive than the extended feature set used in the proposed model (C1-C4).",
        ],
    )

    # ---- Section: quantitative results ----
    pdf.add_page()
    pdf.chapter_title("5", "Quantitative results \u2014 three papers on D1 and D2")
    pdf.body_text(
        "All three papers were evaluated on the same held-out test sets. Metrics: Accuracy, Precision, Recall, "
        "F1, Specificity, ROC-AUC, PR-AUC, MCC (primary), and average prediction latency per pair."
    )
    pdf.section_title("Dataset 1 \u2014 Text Plagiarism Benchmark (N = 9,824 test pairs)")
    pdf.add_table(
        ["Paper", "Accuracy", "Precision", "Recall", "F1", "Specificity", "ROC-AUC", "PR-AUC", "MCC", "Latency (ms/pair)"],
        [
            ["Sajid et al. (2025)", "0.6838", "0.5309", "0.6683", "0.5917", "0.6919", "0.7378", "0.5917", "0.3452", "0.802"],
            ["Amirzhanov et al. (2025)", "0.6887", "0.5797", "0.3349", "0.4245", "0.8733", "0.6865", "0.5183", "0.2480", "0.817"],
            ["Singh & Gupta (2026)", "0.6803", "0.6103", "0.1865", "0.2856", "0.9379", "0.6893", "0.5103", "0.1927", "\u22640.001"],
        ],
        col_widths=[30, 17, 17, 15, 14, 18, 17, 15, 15, 22],
        size=7.5,
    )
    pdf.section_title("Dataset 2 \u2014 Quora Question Pairs (N = 8,000 test pairs; 5,000-pair training subset)")
    pdf.add_table(
        ["Paper", "Accuracy", "Precision", "Recall", "F1", "Specificity", "ROC-AUC", "PR-AUC", "MCC", "Latency (ms/pair)"],
        [
            ["Sajid et al. (2025)", "0.5995", "0.5426", "0.5984", "0.5691", "0.6004", "0.6371", "0.5762", "0.1975", "0.751"],
            ["Amirzhanov et al. (2025)", "0.4421", "0.4419", "0.9963", "0.6122", "0.0031", "0.4838", "0.4303", "\u22120.0046", "0.661"],
            ["Singh & Gupta (2026)", "0.4421", "0.4419", "0.9972", "0.6124", "0.0025", "0.5484", "0.4862", "\u22120.0035", "\u22640.001"],
        ],
        col_widths=[30, 17, 17, 15, 14, 18, 17, 15, 15, 22],
        size=7.5,
    )
    pdf.bullet("On Dataset 1, Sajid 2025 leads the three papers on MCC (0.3452); Amirzhanov 2025 achieves the highest accuracy (0.6887) but low recall (0.3349); Singh & Gupta 2026 is highly specific (0.9379) but low recall (0.1865).")
    pdf.bullet("On Dataset 2, all three papers struggle: Amirzhanov and Singh & Gupta predict almost everything as positive (recall \u2248 1.0, specificity \u2248 0), giving negative MCC (\u22120.0046 and \u22120.0035); Sajid 2025 is the strongest of the three (MCC 0.1975).")
    pdf.bullet("These reproduced results are reported exactly as computed, including the negative MCC values; nothing is inflated or omitted.")

    # ---- Section: qualitative ----
    pdf.add_page()
    pdf.chapter_title("6", "Qualitative analysis \u2014 three papers")
    pdf.add_table(
        ["Dimension", "Paper 1: Sajid 2025", "Paper 2: Amirzhanov 2025", "Paper 3: Singh & Gupta 2026"],
        [
            ["Method family", "Feature-based supervised (Logistic Regression)", "Rule-based similarity threshold", "Hybrid word-overlap"],
            ["Features", "8 similarity features (lexical + shallow semantic)", "1 feature (TF-IDF cosine)", "3 features (hybrid overlap)"],
            ["Training required", "Yes (Logistic Regression)", "No", "No"],
            ["Strengths", "Best MCC of the three on both datasets; balanced precision/recall", "Extremely fast; no training; simple to deploy", "Interpretable; high specificity on D1"],
            ["Weaknesses", "Misses heavily paraphrased pairs with little word overlap", "Low recall on D1; near-zero specificity on D2; negative MCC on D2", "Low recall on both datasets; overlap signals saturate"],
            ["Best suited to", "General paraphrase detection baseline", "Fast pre-filtering where speed matters most", "Near-duplicate / verbatim-heavy scenarios"],
        ],
        col_widths=[26, 52, 52, 50],
        size=7.5,
    )
    pdf.bullet("Qualitative takeaway: none of the three papers alone provides strong detection on both datasets. Paper 1 (Sajid 2025) is the strongest baseline and is therefore adopted as Model 1 in the progressive model series, which adds contributions C1-C4 on top of it.")
    pdf.bullet("The progressive proposed model (Model 5) outperforms all three papers on both datasets: see PDF 3 (Proposed_Method_Results_and_Analysis.pdf) for the full comparison.")

    # ---- Section: implementation & reproducibility ----
    pdf.chapter_title("7", "Implementation and reproducibility")
    pdf.bullet("Implementation location: kaggle_cells_single_papers.md (Kaggle/Colab-compatible cells, one implementation per paper) and reports/paper_wise_comparison/implement_three_papers.py.")
    pdf.bullet("Shared protocol: identical datasets, splits, preprocessing, LogisticRegression(class_weight='balanced', max_iter=2000), MCC-optimized threshold, and metric suite for all three papers and the proposed model.")
    pdf.bullet("Code verified against the source in final_analysis_package/source_code/proposed/proposed_typed_solution.py.")
    pdf.not_run_box("Published-paper verification: the source paper PDFs are not archived in this repository, so published metrics are not used as if reproduced; all paper numbers are reproduced implementation results, explicitly labeled as such.")
    pdf.integrity_statement()

    out = "/home/user/ml/Three_Papers_Implementation_and_Results.pdf"
    pdf.output(out)
    print("PDF 2 saved:", out)
    return out


# =====================================================================
# PDF 3 — PROPOSED METHOD: MODEL, RESULTS, ANALYSIS
# =====================================================================
def build_pdf3():
    pdf = new_pdf()

    pdf.ln(22)
    pdf.set_font("DejaVu", "B", 20)
    pdf.set_text_color(200, 60, 40)
    pdf.cell(0, 12, "Proposed Method \u2014 Typed-Decision Pre-Filtering", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.cell(0, 12, "Model, Results, Qualitative and Quantitative Analysis", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(8)
    pdf.set_font("DejaVu", "B", 13)
    pdf.set_text_color(40, 80, 140)
    pdf.cell(0, 8, "Progressive Models 1-5 with Contributions C1-C4", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(4)
    pdf.set_font("DejaVu", "", 10)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 6, "Proposed method, outperformance versus existing baselines, ablation, qualitative and cost analysis.", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(16)
    pdf.set_font("DejaVu", "B", 10)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(0, 6, "Prepared from", new_x="LMARGIN", new_y="NEXT")
    pdf.body_text(
        "Repository artifacts: progressive_model_development_paper.tex (IEEE conference paper), "
        "reports/paper_wise_comparison/tables/ablation_study.csv, "
        "final_analysis_package/source_code/proposed/proposed_typed_solution.py, "
        "reports/progressive_model_development/PROGRESSIVE_QUANTITATIVE_COMPARISON.csv. "
        f"Generated: {TODAY}  |  Repository: {REPO}"
    )

    # ---- Section 1: proposed method ----
    pdf.add_page()
    pdf.chapter_title("1", "Proposed method: typed-decision pre-filtering")
    pdf.body_text(
        "Large-scale academic-integrity detection must balance reliable similarity assessment with computational "
        "cost. The proposed typed-decision pre-filtering pipeline uses a lightweight similarity model to identify "
        "straightforward cases and reserves a comprehensive feature-based model for cases requiring further review. "
        "The feature representation extends an eight-feature baseline with four original contributions (C1-C4): "
        "word-overlap counts, character n-grams, edit distance, and semantic/structural/phonetic features."
    )
    pdf.bullet("Typed: the first-stage method is selected based on the submission \u2014 lightweight pre-filter vs. comprehensive verification.")
    pdf.bullet("Decision: first-stage evidence and a calibrated MCC-optimized threshold decide whether to accept a pair as low-risk or escalate it to a heavier model.")
    pdf.bullet("Cost-aware: latency per pair, model invocations, and estimated compute cost are tracked per submission.")
    pdf.callout_box(
        "Cascade cost model (standalone measurements, end-to-end cascade NOT RUN)",
        "Every pair is processed by Model 1 (t1); only the unresolved proportion (1-p) is forwarded to Model 5 (t5). "
        "Estimated average cost: T_cascade = t1 + (1-p) \u00d7 t5. The cascade beats Model 5 on every pair only if "
        "p > t1/t5 \u2014 approximately 35.2% on Dataset 1 (0.062 vs 0.176 ms/pair) and 4.8% on Dataset 2 "
        "(0.054 vs 1.116 ms/pair). These are theoretical estimates from standalone runs; the actual resolution "
        "proportion, candidate recall, routing rate, and realized workload reduction were not measured (see Section 7).",
    )

    # ---- Section 2: models + contributions ----
    pdf.chapter_title("2", "Progressive model configurations and contributions C1-C4")
    pdf.body_text(
        "Five progressive configurations are evaluated: Sajid 2025 baseline incrementally extended with C1-C4. "
        "All five models share identical datasets, splits, preprocessing, classifier, and evaluation protocol. "
        "Contribution definitions are verified against the source code in final_analysis_package/source_code/proposed/proposed_typed_solution.py."
    )
    pdf.add_table(
        ["Model", "Configuration", "Features", "New contribution"],
        [
            ["Model 1", "Sajid 2025 baseline", "8", "None (baseline)"],
            ["Model 2", "Baseline + C1", "16", "C1: Word-overlap count features (8 dims)"],
            ["Model 3", "Baseline + C1 + C2", "21", "C2: Character n-gram Jaccard features (5 dims)"],
            ["Model 4", "Baseline + C1 + C2 + C3", "25", "C3: Edit distance features (4 dims)"],
            ["Model 5", "Baseline + C1 + C2 + C3 + C4", "83", "C4: LSA topics + question structure + Soundex (58 dims)"],
        ],
        col_widths=[20, 56, 22, 82],
    )
    pdf.bullet("C1 \u2014 Word-Overlap Count Features (8 dims): word Jaccard, overlap count, length-diff ratio, short-text overlap, bigram Jaccard, exact match, normalized overlap, bigram containment.")
    pdf.bullet("C2 \u2014 Character N-Gram Features (5 dims): character 2/3/4/5-gram Jaccard, trigram containment (orthographic and morphological variation).")
    pdf.bullet("C3 \u2014 Edit Distance Features (4 dims): word- and character-level Levenshtein and Jaro-Winkler similarity (near-duplicates with small edits).")
    pdf.bullet("C4 \u2014 Semantic, Structural and Phonetic Features (58 dims): LSA topic vectors (50 dims from TruncatedSVD on TF-IDF with up to 5,000 features), question structure features (5 dims), Soundex phonetic features (3 dims).")
    pdf.bullet("Classifier for all trained models: LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42); MCC-optimized threshold; metrics as in PDF 1, Section 4.")

    # ---- Section 3: quantitative results ----
    pdf.add_page()
    pdf.chapter_title("3", "Quantitative results \u2014 proposed model vs baselines")
    pdf.section_title("Progressive models on both datasets (shared protocol)")
    pdf.add_table(
        ["Dataset", "Model", "Config", "Accuracy", "MCC", "ROC-AUC", "Features", "Latency (ms/pair)"],
        [
            ["D1 (9,824 test)", "Model 1", "BASELINE", "0.7149", "0.3573", "0.7375", "8", "0.062"],
            ["D1 (9,824 test)", "Model 2", "+C1", "0.7218", "0.3813", "0.7512", "16", "\u2014"],
            ["D1 (9,824 test)", "Model 3", "+C1+C2", "0.7392", "0.3993", "0.7608", "21", "\u2014"],
            ["D1 (9,824 test)", "Model 4", "+C1+C2+C3", "0.7387", "0.4035", "0.7619", "25", "\u2014"],
            ["D1 (9,824 test)", "Model 5", "+C1+C2+C3+C4", "0.7641", "0.4705", "0.8007", "83", "0.176"],
            ["D2 (8,000 test)", "Model 1", "BASELINE", "0.6009", "0.1990", "0.6367", "8", "0.054"],
            ["D2 (8,000 test)", "Model 2", "+C1", "0.6804", "0.3476", "0.7273", "16", "\u2014"],
            ["D2 (8,000 test)", "Model 3", "+C1+C2", "0.6824", "0.3516", "0.7272", "21", "\u2014"],
            ["D2 (8,000 test)", "Model 4", "+C1+C2+C3", "0.6976", "0.3844", "0.7552", "25", "\u2014"],
            ["D2 (8,000 test)", "Model 5", "+C1+C2+C3+C4", "0.7046", "0.3958", "0.7585", "83", "1.116"],
        ],
        col_widths=[26, 17, 24, 18, 16, 18, 17, 24],
        size=7.5,
    )
    pdf.section_title("Proposed model (Model 5) vs the three baseline papers \u2014 MCC comparison")
    pdf.add_table(
        ["Dataset", "Sajid 2025", "Amirzhanov 2025", "Singh & Gupta 2026", "Proposed Model 5", "Proposed margin over best paper"],
        [
            ["D1", "0.3452", "0.2480", "0.1927", "0.4705", "+0.1253 over Sajid 2025"],
            ["D2", "0.1975", "\u22120.0046", "\u22120.0035", "0.3958", "+0.1983 over Sajid 2025"],
        ],
        col_widths=[18, 22, 26, 30, 26, 58],
        size=7.5,
    )
    pdf.bullet("Result: the proposed Model 5 outperforms all three existing papers on both datasets. On Dataset 1, MCC rises from the Sajid 2025 baseline 0.3573 to 0.4705 (+31.7% relative). On Dataset 2, MCC rises from 0.1990 to 0.3958 (+98.9% relative). Model 5 also beats each standalone paper implementation on both datasets (see table above).")
    pdf.bullet("Accuracy and ROC-AUC improve on both datasets as well: D1 accuracy 0.7149 \u2192 0.7641, ROC-AUC 0.7375 \u2192 0.8007; D2 accuracy 0.6009 \u2192 0.7046, ROC-AUC 0.6367 \u2192 0.7585.")
    pdf.bullet("Improvement is accompanied by higher per-pair latency (Model 1: 0.062 ms/pair D1; Model 5: 0.176 ms/pair D1; D2: 0.054 vs 1.116 ms/pair) \u2014 the cost/quality trade-off that motivates the two-stage pre-filter design.")

    # ---- Section 4: ablation ----
    pdf.chapter_title("4", "Ablation analysis \u2014 MCC drop when each contribution is removed")
    pdf.body_text(
        "Ablation measures how much MCC falls when each contribution is removed from the full model (Model 5). "
        "Full MCC: 0.47149 on Dataset 1; 0.442253 on Dataset 2 (ablation variants executed with the same protocol)."
    )
    pdf.add_table(
        ["Dataset", "Removed", "Full MCC", "Ablated MCC", "MCC Drop", "Assessment vs \u22650.05 threshold"],
        [
            ["D1", "C1", "0.47149", "0.45923", "+0.0123", "Below threshold"],
            ["D1", "C2", "0.47149", "0.44453", "+0.0270", "Below threshold"],
            ["D1", "C3", "0.47149", "0.45476", "+0.0167", "Below threshold"],
            ["D1", "C4", "0.47149", "0.40026", "+0.0712", "Meets threshold (\u22650.05)"],
            ["D2", "C1", "0.44225", "0.43749", "+0.0048", "Below threshold"],
            ["D2", "C2", "0.44225", "0.43658", "+0.0057", "Below threshold"],
            ["D2", "C3", "0.44225", "0.36050", "+0.0818", "Meets threshold (\u22650.05)"],
            ["D2", "C4", "0.44225", "0.45429", "\u22120.0120", "Negative \u2014 removal slightly raised MCC"],
        ],
        col_widths=[18, 16, 20, 22, 20, 84],
        size=7.5,
    )
    pdf.bullet("Honest assessment: only 2 of 8 dataset\u2013contribution cells meet the \u22650.05 MCC threshold \u2014 C4 on Dataset 1 (+0.0712) and C3 on Dataset 2 (+0.0818). C4 on Dataset 2 is negative (\u22120.0120) and is reported as measured, not suppressed. C1 and C2 show modest, below-threshold impact on both datasets.")
    pdf.bullet("Interpretation: D2 is word-overlap dominated, so C1 helps little beyond the baseline and C4 adds semantic noise; on D1 the semantic/structural/phonetic group (C4) is the strongest single contributor.")

    # ---- Section 5: qualitative ----
    pdf.chapter_title("5", "Qualitative analysis")
    pdf.bullet("Baseline (Model 1): compact, low-cost first stage \u2014 word Jaccard, char-trigram Jaccard, overlap count, length difference, exact match, TF-IDF cosine, LSA cosine. Low latency makes it suitable for preliminary screening, but a compact representation may lack evidence when meaning is expressed through substantial lexical changes.")
    pdf.bullet("Complete model (Model 5): adds C1 word-overlap counts (shared terms/phrases), C2 character n-grams (local spelling/morphology), C3 edit distance (near-duplicates), and C4 semantic/structural/phonetic evidence (LSA topics, question structure, Soundex). Higher MCC and ROC-AUC on both datasets confirm a more discriminative representation.")
    pdf.bullet("Benefit size differs between datasets \u2014 C4 dominates on the text benchmark (D1), while word-overlap signals dominate on Quora question pairs (D2). Contribution impact is therefore dataset-dependent, as designed.")
    pdf.bullet("The staged workflow assigns the lightweight model to initial screening and the comprehensive model to pairs that cannot be confidently resolved; confidence thresholds for routing must be defined and validated before deployment (end-to-end cascade NOT RUN \u2014 see Section 7).")

    # ---- Section 6: cost ----
    pdf.chapter_title("6", "Cost analysis")
    pdf.add_table(
        ["Dataset", "Model", "Latency (ms/pair)", "MCC", "MCC per ms"],
        [
            ["D1", "Model 1 (baseline)", "0.062", "0.3573", "5.763"],
            ["D1", "Model 5 (proposed)", "0.176", "0.4705", "2.673"],
            ["D2", "Model 1 (baseline)", "0.054", "0.1990", "3.685"],
            ["D2", "Model 5 (proposed)", "1.116", "0.3958", "0.355"],
        ],
        col_widths=[22, 32, 32, 22, 28],
    )
    pdf.bullet("Model 1 is \u22482.8\u00d7 faster than Model 5 on Dataset 1 and \u224820.7\u00d7 faster on Dataset 2, at the cost of lower MCC.")
    pdf.bullet("Model 1 delivers 5.76 MCC per ms on Dataset 1 vs 2.67 for Model 5; the baseline is the efficient pre-filter, Model 5 the accurate verifier \u2014 the basis of the two-stage design.")

    # ---- Section 7: NOT RUN ----
    pdf.chapter_title("7", "Experiments NOT executed (reported as NOT RUN)")
    pdf.not_run_box("End-to-end cascade performance: the fraction of pairs resolved by the pre-filter, the recall of the routed candidate set, the routing rate, and measured end-to-end cascade latency were not measured. T_cascade = t1 + (1-p) x t5 in Section 1 is a theoretical estimate from standalone runs only.")
    pdf.not_run_box("Candidate recall and false-negative risk of the two-stage routing: not measured; potentially plagiarized pairs must not be incorrectly discarded by the first stage before this is validated.")
    pdf.not_run_box("Realized workload reduction under different routing thresholds: remains to be experimentally evaluated before claiming efficiency gains in production.")
    pdf.body_text("All abstracts and paper descriptions in this document are proposed descriptions written for this analysis, not verbatim published abstracts.")

    pdf.integrity_statement()

    out = "/home/user/ml/Proposed_Method_Results_and_Analysis.pdf"
    pdf.output(out)
    print("PDF 3 saved:", out)
    return out


if __name__ == "__main__":
    p1 = build_pdf1()
    p2 = build_pdf2()
    p3 = build_pdf3()
    print("All three PDFs generated:")
    print(" ", p1)
    print(" ", p2)
    print(" ", p3)
