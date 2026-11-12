# Dataset analysis summary

- Dataset 1 (text): SNLI/Pereira "MIT Plagiarism Detection Dataset" re-export; premise-grouped
  partitions; zero cross-split premise/pair overlap verified (see DATASET_QUALITY_REPORT.json).
- Dataset 2 (code): IR-Plag Java corpus, 467 physical .java files on disk, policies A (720
  pairs) / B (709); positives carry obfuscation severity L1-L6; L6 is all-positive by construction.
- Class balance is moderate (text ~34% positive; code ~47%); no missing rows; duplicate-collision
  rows handled by audited policies upstream.
- Figures: `08_VISUALIZATIONS/DATASET_DISTRIBUTION/figure_2_distributions.{png,pdf}`.
- Quantitative files: DATASET1_ANALYSIS.csv, DATASET2_ANALYSIS.csv, DATASET_COMPARISON.csv.
