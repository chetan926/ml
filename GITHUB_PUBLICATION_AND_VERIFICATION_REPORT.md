# GitHub Publication and Verification Report

**Project Title:** An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection  
**Repository Target:** `https://github.com/chetan926/ml.git`  
**GitHub Owner:** `chetan926`  
**Local Working Directory:** `D:\ml rearcch project ad`  
**Report Date:** November 12, 2026  
**Final Verification Verdict:** `PUBLICATION_BLOCKED`  

---

## 1. Executive Publication Status

| Parameter | Current Status / Value |
| :--- | :--- |
| **Repository URL** | `https://github.com/chetan926/ml.git` |
| **Publication Status** | **PUBLICATION_BLOCKED** (Remote authentication barrier; local repository fully staged and committed) |
| **Target Branch** | `main` (symref verified on remote: `refs/heads/main`) |
| **Latest Commit Hash** | `52917834064c7f2127fe3db7e7187d5bc9e675e9` |
| **Commit Message** | `Complete Research Framework Documentation, Licensing, Packaging, and Phase 4.1 Verification` |
| **Total Tracked Files** | **992 files** staged and committed into branch `main` |
| **Remote Repository State** | Verified via Dulwich client: `{b'HEAD': None, symref: b'refs/heads/main'}` (Repository exists on GitHub and is currently empty) |
| **Authentication Barrier** | `HTTPUnauthorized: No valid credentials provided` (GitHub requires write authentication) |

---

## 2. Local-Versus-Remote Inventory Comparison

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                          REPOSITORY SYNCHRONIZATION STATUS                             │
├──────────────────────────┬───────────────────────────┬─────────────────────────────────┤
│ Local Working Directory  │ Local Git Repository      │ Remote: chetan926/ml            │
│ 588 non-ignored files    │ 992 tracked files         │ 0 commits (Empty repo)          │
│ 100% frozen baseline     │ HEAD: 5291783 (main)      │ HEAD: None (refs/heads/main)    │
│ Phase 4.1 Verified       │ Ready for immediate push  │ Awaiting auth token / SSH key   │
└──────────────────────────┴───────────────────────────┴─────────────────────────────────┘
```

- **Remote Inspection Result:** An active HTTPS probe to `https://github.com/chetan926/ml.git/info/refs?service=git-receive-pack` confirms the remote repository is valid, active, and contains 0 existing commits (`HEAD: None`). No remote history will be overwritten.
- **Local Git State:** Initialized with branch `main`, tracking 992 files across all research phases, baseline modules, tests, datasets, artifacts, and documentation.

---

## 3. Four Research Contributions & Repository Locations

| Contribution | Implementation Module | Verification / Artifact Location |
| :--- | :--- | :--- |
| **C1: Multi-View Feature Fusion** | • `research_upgrade/src/features/feature_pipeline.py`<br>• `research_upgrade/src/features/text_lexical.py`<br>• `research_upgrade/src/features/text_character.py`<br>• `research_upgrade/src/features/text_semantic.py`<br>• `research_upgrade/src/features/text_structural.py`<br>• `research_upgrade/src/features/code_ast_features.py` | • `research_upgrade/artifacts/phase_3/phase_3_3/features/`<br>• `research_upgrade/phase_4_1_corrections/artifacts/PHASE_4_1_ABLATION_RESULTS.csv` |
| **C2: Uncertainty & Group Conformal** | • `research_upgrade/src/calibration/calibrators.py`<br>• `research_upgrade/src/calibration/uncertainty.py`<br>• `research_upgrade/phase_4_1_corrections/src/calibration/group_conformal.py` | • `research_upgrade/phase_4_1_corrections/artifacts/PHASE_4_1_CALIBRATION_METRICS.csv`<br>• `research_upgrade/phase_4_1_corrections/artifacts/figures/figure_1_calibration_curves.png`<br>• `figure_2_risk_coverage.png` |
| **C3: Budget-Adaptive Routing & Cost** | • `research_upgrade/src/routing/router.py`<br>• `research_upgrade/src/routing/stage2_verifier.py`<br>• `research_upgrade/src/routing/cost_model.py`<br>• `research_upgrade/src/routing/pareto.py`<br>• `research_upgrade/phase_4_1_corrections/src/routing/code_group_evaluator.py`<br>• `research_upgrade/phase_4_1_corrections/src/evaluation/profiling.py` | • `research_upgrade/phase_4_1_corrections/artifacts/PHASE_4_1_COST_ANALYSIS.csv`<br>• `research_upgrade/phase_4_1_corrections/artifacts/PHASE_4_1_PARETO_OPERATING_POINTS.csv`<br>• `research_upgrade/phase_4_1_corrections/artifacts/figures/figure_3_pareto_cost_recall.png` |
| **C4: Explainable Reviewer Dossiers** | • `research_upgrade/phase_4_1_corrections/src/explainability/evidence_extractor.py`<br>• `research_upgrade/phase_4_1_corrections/src/explainability/dossier_generator.py` | • `research_upgrade/phase_4_1_corrections/artifacts/PHASE_4_1_RESEARCH_RESULTS.json` (sample dossiers)<br>• Reviewer schema `4.1.0_OPERATIONAL_REVIEW` |

---

## 4. Key Experimental Results Committed

### Text Modality (3,000 Held-Out Pairs, 1,015 Premise Groups)
- **Proposed Framework:** Accuracy = **0.7893**, Precision = **0.7234**, Recall = **0.6218**, F1 = **0.6688**, MCC = **0.5188** (95% CI $[0.4851, 0.5519]$).
- **Dense Semantic Baseline:** Accuracy = 0.6987, Precision = 0.5369, Recall = 0.8645, F1 = 0.6624, MCC = 0.4549.
- **Paired Superiority:** $\Delta \text{MCC} = \mathbf{+0.0638}$ (Statistically Significant, $p < 10^{-16}$). Eliminates **521 false positives** (lowering FP from 765 to 244).
- **Stage-2 Heavy Verifier Workload Avoidance:** **47.23%** (1,417 / 3,000 pairs decided at Stage-1).
- **Conformal Coverage:** Empirical group coverage = **93.69%**, pair coverage = **97.80%**, average set size = **1.6080**.

### Code Modality (IR-Plag Java Corpus, 720 Policy A Pairs)
- **Nested LOGO (Zero Leakage):** Accuracy = **0.8486**, Precision = **0.8157**, Recall = **0.8801**, F1 = **0.8467**, MCC = **0.6996**, Stage-2 avoided = **15.14%** (109 / 720).
- **Stratified 5-Fold (Naive Split):** Accuracy = 0.9111, Precision = 0.9137, Recall = 0.8977, F1 = 0.9056, MCC = 0.8217, Stage-2 avoided = 29.72%.
- **Controlled Leakage Inflation:** $\Delta \text{MCC} = \mathbf{+0.1221}$ artificial gain from cross-fold program sharing.
- **Program Overlap Verification:** 0 reference programs and 0 suspicious submissions shared across assignment cases (SHA-256 verified).

---

## 5. File Exclusions & Security Governance

The `.gitignore` strictly excludes:
- `.venv/`, `venv/`, `env/` (Virtual environments and site-packages)
- `uv.exe` and binary executables (`*.exe`, `*.dll`, `*.pyd`)
- `__pycache__/`, `*.pyc`, `*.pyo` (Compiled bytecode)
- `.matplotlib/`, `matplotlib_cache/` (Local UI/rendering caches)
- `scratch/`, `tmp/`, `temp/` (Ad-hoc testing scratchpads)
- Operating system files (`Thumbs.db`, `.DS_Store`, `desktop.ini`)

**Secrets Audit:** Zero personal access tokens, private keys, passwords, or cloud credentials exist in the committed files, git history, or documentation.

---

## 6. Authentication Blocker Diagnosis

When executing the Git push via the transport layer:
```
Target: https://github.com/chetan926/ml.git
Branch: refs/heads/main
Transport: HTTPS
Error: dulwich.errors.HTTPUnauthorized: No valid credentials provided
```

### Root Cause Analysis:
1. **GitHub CLI (`gh.exe`)**: Not installed on the system PATH.
2. **Git Executable (`git.exe`)**: Not installed on the system PATH or standard Program Files locations.
3. **Windows Credential Manager**: Empty (`* NONE *` credentials stored).
4. **SSH Configuration (`~/.ssh`)**: No SSH key pair exists in the user profile directory.
5. **Security Guideline Compliance**: In accordance with Phase 5 & 6 instructions (*"Do not ask me to paste my GitHub token into ChatGPT or commit credentials. If authentication or permissions prevent publication, report the exact blocker"*), no credentials were hardcoded or exposed.

---

## 7. Resolution & Single-Step Unblocking Command

Because the local repository is 100% prepared, staged, committed, and tested, publication to GitHub requires only providing credentials to complete the HTTPS transmission.

### Method 1: Using GitHub CLI (Recommended if installed locally)
```bash
gh auth login
git push -u origin main
```

### Method 2: Single-Line Push via Python Dulwich (No Git Installation Required)
In PowerShell within the project directory, run:
```powershell
.venv\Scripts\python.exe -c "from dulwich import porcelain; from dulwich.repo import Repo; repo = Repo('.'); porcelain.push(repo, 'https://<GITHUB_PERSONAL_ACCESS_TOKEN>@github.com/chetan926/ml.git', 'refs/heads/main'); print('SUCCESS: Published to https://github.com/chetan926/ml.git')"
```

---

## 8. Final Verdict

**Verdict:** `PUBLICATION_BLOCKED`  
*All research contributions, experimental data, unit tests, independent verification evidence, Apache-2.0 licensing, and documentation are committed locally in commit `52917834064c7f2127fe3db7e7187d5bc9e675e9`. Remote transmission is held pending GitHub authorization credentials.*
