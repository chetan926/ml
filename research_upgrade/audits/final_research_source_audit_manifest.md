# Final Research Source Code and Evaluation Audit Manifest

**Project:** An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection  
**Package Name:** `final_research_source_audit.zip`  
**Generated Timestamp:** 2026-11-11 16:16:00  
**Total Files Included:** 79  
**Declared Missing Count:** 2  

---

## 1. Declared Missing Modules / Files

| Requested Component | Status | Technical Explanation |
| :--- | :---: | :--- |
| `research_upgrade/src/evaluation/benchmarking.py` | **MISSING** | No standalone benchmarking.py module was designed or created. Benchmarking functionality is implemented in research_upgrade/src/data/text_loader.py (TextBenchmarkDataLoader), code_loader.py (CodeBenchmarkDataLoader), and research_upgrade/audits/final_research/run_final_research_pipeline.py. |
| `pair_level_predictions.csv` | **MISSING** | No standalone per-pair tabular predictions CSV was serialized to disk during the final research pipeline pass. Complete aggregated prediction metrics, conformal group metrics, and selective prediction statistics are fully captured in FINAL_RESEARCH_RESULTS.json and FINAL_CLASSIFICATION_METRICS.csv, while group identifiers and partition mappings are in text_partition_membership.csv. |

---

## 2. Included Source, Test, and Artifact Files

| Category | File Path | Size (Bytes) | SHA-256 Checksum |
| :--- | :--- | :---: | :--- |
| 1. Integrated Pipelines | [`research_upgrade/src/pipeline/integrated_framework.py`](file:///research_upgrade/src/pipeline/integrated_framework.py) | 17,742 | `c675a37102002fec...` |
| 1. Integrated Pipelines | [`research_upgrade/src/pipeline/__init__.py`](file:///research_upgrade/src/pipeline/__init__.py) | 247 | `e2aa013b1121a8b4...` |
| 2. Calibration & Uncertainty | [`research_upgrade/src/calibration/calibrators.py`](file:///research_upgrade/src/calibration/calibrators.py) | 6,167 | `37367ef1453d4fb8...` |
| 2. Calibration & Uncertainty | [`research_upgrade/src/calibration/uncertainty.py`](file:///research_upgrade/src/calibration/uncertainty.py) | 4,180 | `d610b4c9088ced76...` |
| 2. Calibration & Uncertainty | [`research_upgrade/src/calibration/calibration_metrics.py`](file:///research_upgrade/src/calibration/calibration_metrics.py) | 5,066 | `25c34c960993f6df...` |
| 2. Calibration & Uncertainty | [`research_upgrade/src/calibration/__init__.py`](file:///research_upgrade/src/calibration/__init__.py) | 986 | `76608eb331ab90b3...` |
| 3. Routing, Verifiers, Cost & Pareto | [`research_upgrade/src/routing/router.py`](file:///research_upgrade/src/routing/router.py) | 11,134 | `04c1ae0ad8a44041...` |
| 3. Routing, Verifiers, Cost & Pareto | [`research_upgrade/src/routing/stage2_verifier.py`](file:///research_upgrade/src/routing/stage2_verifier.py) | 5,616 | `1af8196243dd4e16...` |
| 3. Routing, Verifiers, Cost & Pareto | [`research_upgrade/src/routing/cost_model.py`](file:///research_upgrade/src/routing/cost_model.py) | 6,000 | `627346ae333cb7d3...` |
| 3. Routing, Verifiers, Cost & Pareto | [`research_upgrade/src/routing/pareto.py`](file:///research_upgrade/src/routing/pareto.py) | 3,657 | `a34437bbab474fd6...` |
| 3. Routing, Verifiers, Cost & Pareto | [`research_upgrade/src/routing/decision_states.py`](file:///research_upgrade/src/routing/decision_states.py) | 1,635 | `89121fe764db92ca...` |
| 3. Routing, Verifiers, Cost & Pareto | [`research_upgrade/src/routing/__init__.py`](file:///research_upgrade/src/routing/__init__.py) | 844 | `38aa62c60b3d53fc...` |
| 4. Group Conformal Prediction | [`research_upgrade/src/calibration/group_conformal.py`](file:///research_upgrade/src/calibration/group_conformal.py) | 6,889 | `6d962b61510bc309...` |
| 5. Evaluation, Metrics, Testing & Ablation Modules | [`research_upgrade/src/evaluation/statistical_testing.py`](file:///research_upgrade/src/evaluation/statistical_testing.py) | 7,858 | `b43e166e4f641ce6...` |
| 5. Evaluation, Metrics, Testing & Ablation Modules | [`research_upgrade/src/evaluation/feature_cost.py`](file:///research_upgrade/src/evaluation/feature_cost.py) | 2,733 | `28ea3f77df48dfed...` |
| 5. Evaluation, Metrics, Testing & Ablation Modules | [`research_upgrade/src/evaluation/feature_quality.py`](file:///research_upgrade/src/evaluation/feature_quality.py) | 3,171 | `be68d007b18e05ed...` |
| 5. Evaluation, Metrics, Testing & Ablation Modules | [`research_upgrade/src/evaluation/__init__.py`](file:///research_upgrade/src/evaluation/__init__.py) | 336 | `d50c44d287bdc62e...` |
| 5. Evaluation, Metrics, Testing & Ablation Modules | [`research_upgrade/src/candidate_generation/ablation_experiments.py`](file:///research_upgrade/src/candidate_generation/ablation_experiments.py) | 7,792 | `65bf1af67b29140c...` |
| 5. Evaluation, Metrics, Testing & Ablation Modules | [`research_upgrade/src/candidate_generation/candidate_evaluation.py`](file:///research_upgrade/src/candidate_generation/candidate_evaluation.py) | 3,797 | `a5d8005513e008bb...` |
| 5. Evaluation, Metrics, Testing & Ablation Modules | [`research_upgrade/src/candidate_generation/robustness_analysis.py`](file:///research_upgrade/src/candidate_generation/robustness_analysis.py) | 5,378 | `93705832be67a7b3...` |
| 5. Evaluation, Metrics, Testing & Ablation Modules | [`research_upgrade/src/candidate_generation/threshold_selection.py`](file:///research_upgrade/src/candidate_generation/threshold_selection.py) | 5,436 | `66cd6869091d2baf...` |
| 5. Evaluation, Metrics, Testing & Ablation Modules | [`research_upgrade/src/candidate_generation/score_fusion.py`](file:///research_upgrade/src/candidate_generation/score_fusion.py) | 9,686 | `914d58767c74b1ea...` |
| 5. Evaluation, Metrics, Testing & Ablation Modules | [`research_upgrade/src/candidate_generation/text_candidate_scorer.py`](file:///research_upgrade/src/candidate_generation/text_candidate_scorer.py) | 8,652 | `f2c18c7cfc658c25...` |
| 5. Evaluation, Metrics, Testing & Ablation Modules | [`research_upgrade/src/candidate_generation/code_candidate_scorer.py`](file:///research_upgrade/src/candidate_generation/code_candidate_scorer.py) | 9,019 | `c7ebd00befd26b76...` |
| 5. Evaluation, Metrics, Testing & Ablation Modules | [`research_upgrade/src/candidate_generation/__init__.py`](file:///research_upgrade/src/candidate_generation/__init__.py) | 1,796 | `4a88b1418e656ef4...` |
| 5. Evaluation, Metrics, Testing & Ablation Modules | [`research_upgrade/src/explainability/evidence_metrics.py`](file:///research_upgrade/src/explainability/evidence_metrics.py) | 3,026 | `dd3feb6907e87042...` |
| 5. Evaluation, Metrics, Testing & Ablation Modules | [`research_upgrade/src/explainability/evidence_extractor.py`](file:///research_upgrade/src/explainability/evidence_extractor.py) | 7,863 | `57e79b0f22b3963f...` |
| 5. Evaluation, Metrics, Testing & Ablation Modules | [`research_upgrade/src/explainability/dossier_generator.py`](file:///research_upgrade/src/explainability/dossier_generator.py) | 2,632 | `79f3077a0f01451f...` |
| 5. Evaluation, Metrics, Testing & Ablation Modules | [`research_upgrade/src/explainability/__init__.py`](file:///research_upgrade/src/explainability/__init__.py) | 527 | `4a8a67cc863b7db0...` |
| 5. Evaluation, Metrics, Testing & Ablation Modules | [`research_upgrade/src/features/feature_pipeline.py`](file:///research_upgrade/src/features/feature_pipeline.py) | 10,155 | `ab32c8d441569203...` |
| 5. Evaluation, Metrics, Testing & Ablation Modules | [`research_upgrade/src/features/text_lexical.py`](file:///research_upgrade/src/features/text_lexical.py) | 3,646 | `b811e3dce9ea82f0...` |
| 5. Evaluation, Metrics, Testing & Ablation Modules | [`research_upgrade/src/features/text_character.py`](file:///research_upgrade/src/features/text_character.py) | 4,454 | `733597cee201b082...` |
| 5. Evaluation, Metrics, Testing & Ablation Modules | [`research_upgrade/src/features/text_semantic.py`](file:///research_upgrade/src/features/text_semantic.py) | 4,679 | `be8fab10bdcf4dab...` |
| 5. Evaluation, Metrics, Testing & Ablation Modules | [`research_upgrade/src/features/text_structural.py`](file:///research_upgrade/src/features/text_structural.py) | 3,071 | `b3071b849b7f2252...` |
| 5. Evaluation, Metrics, Testing & Ablation Modules | [`research_upgrade/src/features/code_token.py`](file:///research_upgrade/src/features/code_token.py) | 2,940 | `a83fe128baeb671d...` |
| 5. Evaluation, Metrics, Testing & Ablation Modules | [`research_upgrade/src/features/code_identifier.py`](file:///research_upgrade/src/features/code_identifier.py) | 3,536 | `58c6ccfed832e0de...` |
| 5. Evaluation, Metrics, Testing & Ablation Modules | [`research_upgrade/src/features/code_ast.py`](file:///research_upgrade/src/features/code_ast.py) | 6,301 | `7fffc50befa839a7...` |
| 5. Evaluation, Metrics, Testing & Ablation Modules | [`research_upgrade/src/features/code_complexity.py`](file:///research_upgrade/src/features/code_complexity.py) | 4,201 | `d0fa73ac05c12b20...` |
| 5. Evaluation, Metrics, Testing & Ablation Modules | [`research_upgrade/src/features/__init__.py`](file:///research_upgrade/src/features/__init__.py) | 1,241 | `33797393165a3567...` |
| 5. Evaluation, Metrics, Testing & Ablation Modules | [`research_upgrade/src/data/text_loader.py`](file:///research_upgrade/src/data/text_loader.py) | 9,039 | `d5d97baf08d4e23b...` |
| 5. Evaluation, Metrics, Testing & Ablation Modules | [`research_upgrade/src/data/code_loader.py`](file:///research_upgrade/src/data/code_loader.py) | 10,659 | `b102f7da7ab4c6da...` |
| 5. Evaluation, Metrics, Testing & Ablation Modules | [`research_upgrade/src/data/partition_validation.py`](file:///research_upgrade/src/data/partition_validation.py) | 5,221 | `517f6c0761f717b2...` |
| 5. Evaluation, Metrics, Testing & Ablation Modules | [`research_upgrade/src/data/schema_validation.py`](file:///research_upgrade/src/data/schema_validation.py) | 7,760 | `b40f5e39522eb4f2...` |
| 5. Evaluation, Metrics, Testing & Ablation Modules | [`research_upgrade/src/data/__init__.py`](file:///research_upgrade/src/data/__init__.py) | 623 | `ea58d0d9adc0592e...` |
| 6. Configuration Files | [`research_upgrade/src/config.py`](file:///research_upgrade/src/config.py) | 6,737 | `98652db7f9ffb702...` |
| 6. Configuration Files | [`research_upgrade/src/logging_utils.py`](file:///research_upgrade/src/logging_utils.py) | 4,062 | `de839c7eb7f34a9a...` |
| 6. Configuration Files | [`research_upgrade/src/governance.py`](file:///research_upgrade/src/governance.py) | 5,698 | `1395ebb1db47f110...` |
| 6. Configuration Files | [`research_upgrade/src/__init__.py`](file:///research_upgrade/src/__init__.py) | 411 | `c2b5a23eb6635393...` |
| 7. Automated Test Suites | [`tests/test_integrated_pipeline.py`](file:///tests/test_integrated_pipeline.py) | 3,254 | `ace66642bae641ec...` |
| 7. Automated Test Suites | [`tests/test_calibration.py`](file:///tests/test_calibration.py) | 4,129 | `b251b4beaadc1a69...` |
| 7. Automated Test Suites | [`tests/test_routing.py`](file:///tests/test_routing.py) | 4,235 | `46190a9cfb9b1b42...` |
| 7. Automated Test Suites | [`tests/test_statistical_testing.py`](file:///tests/test_statistical_testing.py) | 2,203 | `563be26d8b4ea16b...` |
| 7. Automated Test Suites | [`tests/test_explainability.py`](file:///tests/test_explainability.py) | 3,389 | `d4420c84c247f2e8...` |
| 7. Automated Test Suites | [`tests/test_candidate_evaluation.py`](file:///tests/test_candidate_evaluation.py) | 2,756 | `aa2471ce550b65d3...` |
| 7. Automated Test Suites | [`tests/test_threshold_selection.py`](file:///tests/test_threshold_selection.py) | 2,261 | `e77c4c0772063f47...` |
| 7. Automated Test Suites | [`tests/test_score_fusion.py`](file:///tests/test_score_fusion.py) | 2,828 | `ae855e6443699d40...` |
| 7. Automated Test Suites | [`tests/test_candidate_reproducibility.py`](file:///tests/test_candidate_reproducibility.py) | 2,625 | `2b9f684bdd112cff...` |
| 7. Automated Test Suites | [`tests/test_text_candidate_generation.py`](file:///tests/test_text_candidate_generation.py) | 2,994 | `bd7059776a5f14c6...` |
| 7. Automated Test Suites | [`tests/test_code_candidate_generation.py`](file:///tests/test_code_candidate_generation.py) | 2,887 | `357d6b8af493ca36...` |
| 7. Automated Test Suites | [`tests/test_feature_pipeline.py`](file:///tests/test_feature_pipeline.py) | 6,219 | `c7de936c08096254...` |
| 7. Automated Test Suites | [`tests/test_text_features.py`](file:///tests/test_text_features.py) | 5,049 | `a4df26e574bb8132...` |
| 7. Automated Test Suites | [`tests/test_code_features.py`](file:///tests/test_code_features.py) | 7,239 | `8dcd6ec01360082b...` |
| 7. Automated Test Suites | [`tests/test_feature_reproducibility.py`](file:///tests/test_feature_reproducibility.py) | 4,353 | `32a706a008972bbc...` |
| 7. Automated Test Suites | [`tests/test_rkrgst.py`](file:///tests/test_rkrgst.py) | 5,744 | `4710667146844dbd...` |
| 7. Automated Test Suites | [`tests/test_data_loaders.py`](file:///tests/test_data_loaders.py) | 4,182 | `31b6f4b2fae341a0...` |
| 7. Automated Test Suites | [`tests/test_leakage_protection.py`](file:///tests/test_leakage_protection.py) | 7,990 | `462185a07bc93f89...` |
| 7. Automated Test Suites | [`tests/test_governance.py`](file:///tests/test_governance.py) | 4,703 | `60589192684ce2fd...` |
| 7. Automated Test Suites | [`tests/test_schema_validation.py`](file:///tests/test_schema_validation.py) | 3,259 | `4ca241553b6a1586...` |
| 8. Final Evaluation Outputs & Group Identifiers | [`research_upgrade/artifacts/final_research/FINAL_RESEARCH_RESULTS.json`](file:///research_upgrade/artifacts/final_research/FINAL_RESEARCH_RESULTS.json) | 29,030 | `666d58c99ca73ade...` |
| 8. Final Evaluation Outputs & Group Identifiers | [`research_upgrade/artifacts/final_research/FINAL_CLASSIFICATION_METRICS.csv`](file:///research_upgrade/artifacts/final_research/FINAL_CLASSIFICATION_METRICS.csv) | 1,844 | `ae563f716bdfb324...` |
| 8. Final Evaluation Outputs & Group Identifiers | [`research_upgrade/artifacts/final_research/FINAL_CALIBRATION_METRICS.csv`](file:///research_upgrade/artifacts/final_research/FINAL_CALIBRATION_METRICS.csv) | 8,149 | `085588d061b490ed...` |
| 8. Final Evaluation Outputs & Group Identifiers | [`research_upgrade/artifacts/final_research/FINAL_STATISTICAL_ANALYSIS.json`](file:///research_upgrade/artifacts/final_research/FINAL_STATISTICAL_ANALYSIS.json) | 2,984 | `65388e17fcb21dd8...` |
| 8. Final Evaluation Outputs & Group Identifiers | [`research_upgrade/artifacts/final_research/FINAL_BASELINE_COMPARISON.csv`](file:///research_upgrade/artifacts/final_research/FINAL_BASELINE_COMPARISON.csv) | 665 | `da2822c5a669248f...` |
| 8. Final Evaluation Outputs & Group Identifiers | [`research_upgrade/artifacts/final_research/FINAL_COST_ANALYSIS.csv`](file:///research_upgrade/artifacts/final_research/FINAL_COST_ANALYSIS.csv) | 733 | `e7d414993cfc7c1a...` |
| 8. Final Evaluation Outputs & Group Identifiers | [`research_upgrade/artifacts/final_research/FINAL_ROBUSTNESS_RESULTS.csv`](file:///research_upgrade/artifacts/final_research/FINAL_ROBUSTNESS_RESULTS.csv) | 378 | `6feb7832476764f4...` |
| 8. Final Evaluation Outputs & Group Identifiers | [`research_upgrade/artifacts/final_research/FINAL_ABLATION_RESULTS.csv`](file:///research_upgrade/artifacts/final_research/FINAL_ABLATION_RESULTS.csv) | 946 | `2c43377c2c2020b8...` |
| 8. Final Evaluation Outputs & Group Identifiers | [`research_upgrade/audits/final_research/run_final_research_pipeline.py`](file:///research_upgrade/audits/final_research/run_final_research_pipeline.py) | 58,093 | `9793e1890328324e...` |
| 8. Final Evaluation Outputs & Group Identifiers | [`research_upgrade/audits/phase_2_3_5/text_partition_membership.csv`](file:///research_upgrade/audits/phase_2_3_5/text_partition_membership.csv) | 897,904 | `f18639813f65d9d6...` |
| 8. Final Evaluation Outputs & Group Identifiers | [`research_upgrade/artifacts/phase_3/phase_3_2/phase_3_2_partition_membership_audit.json`](file:///research_upgrade/artifacts/phase_3/phase_3_2/phase_3_2_partition_membership_audit.json) | 2,777 | `486cada5ecb2ed2d...` |
