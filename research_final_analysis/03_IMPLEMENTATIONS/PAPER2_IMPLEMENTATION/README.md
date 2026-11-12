# Paper 2 implementation — Gandhi et al. (2024)

Realization: frozen `CodeBaselineSuiteLOGO` B-C3 (RBF-SVM, C=1, gamma=scale) under nested
LOGO over CODE_SVM_FEATURES; identical-seed shadow refit with decision_function reproduces
hard predictions bit-exactly (integrity proof in IMPLEMENTATION_INDEX.json).
Deviation (audited): Python→Java cross-language adaptation.
Evidence: `../../04_EXPERIMENTAL_RESULTS/PREDICTIONS/B-C3_code_predictions.csv`.
