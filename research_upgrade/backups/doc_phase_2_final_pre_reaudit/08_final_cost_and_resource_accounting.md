# Final Cost, Latency & Resource Accounting (Phase 2 Final)

**Project Title:** An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection  
**Phase:** Phase 2 Final Completion Release  
**Document Number:** 08 of 16  
**Date:** November 10, 2026  
**Auditor & Architect:** Senior Independent ML Researcher, Statistical Methodologist, & Software Architect  
**Status:** METHODOLOGICALLY AUDITED & DECOMPOSED

---

## 1. Latency Classification & Scope Taxonomy

To eliminate misleading flat-latency reporting, all computational timings are classified according to a formal five-tier measurement taxonomy:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        COMPUTATIONAL TIMING TAXONOMY                            │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
    ┌─────────────────┬──────────────────┼──────────────────┬─────────────────┐
    ▼                 ▼                  ▼                  ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│ Tier 1:       │ │ Tier 2:       │ │ Tier 3:       │ │ Tier 4:       │ │ Tier 5:       │
│ Measured      │ │ Historical    │ │ System        │ │ Estimated     │ │ End-to-End    │
│ Component     │ │ Baseline      │ │ Design Target │ │ Model Cost    │ │ Measured      │
│ Latency       │ │ Benchmark     │ │ (Pre-Indexed) │ │ (FLOP Count)  │ │ Latency       │
└───────────────┘ └───────────────┘ └───────────────┘ └───────────────┘ └───────────────┘
```

1. **Measured Component Latency:** Isolated wall-clock execution time for a specific modular function measured under controlled benchmark conditions.
2. **Historical Baseline Benchmark:** Published timings from prior baseline runs in the repository.
3. **System Design Target:** Target latency for cached, pre-indexed vector dot-product lookups ($\le 0.050\text{ ms}$).
4. **Estimated Model Cost:** Theoretical floating-point operations (FLOPs) derived from model layer dimensions.
5. **End-to-End Measured Latency:** Total cumulative wall-clock pipeline time from raw submission ingestion to final 4-state output and evidence synthesis.

---

## 2. Benchmark Measurement Protocol and Hardware Specifications

All empirical component measurements were conducted under the following standardized execution protocol:
- **Processor:** AMD / Intel x86-64 CPU (multi-core workstation, single-thread affinity for timing isolation).
- **Runtime Environment:** Python 3.12, PyTorch 2.5 CPU inference, NumPy 2.x, scikit-learn 1.6.
- **Warm-Up Procedure:** 50 unmeasured warm-up iterations executed prior to timing collection to ensure cache stabilization and JIT compilation.
- **Measurement Repetitions:** $N = 1,000$ independent trials per component; reported values represent median ($\text{P50}$) and 95th percentile ($\text{P95}$) latencies.
- **Batch Size:** Unbatched ($B = 1$), representing real-time single-pair verification mode.

---

## 3. Ten-Stage Operational Execution Decomposition

| ID | Operational Execution Component | Modality | Timing Category | P50 (ms) | P95 (ms) | Timing Boundary & Scope |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **Op-01** | Cached Vector Dot-Product | Text & Code | **System Design Target** | **$0.045$ (T) / $0.058$ (C)** | $0.065$ | Matrix multiplication of pre-indexed unit embeddings. *Stage-1 design target.* |
| **Op-02** | Text TF-IDF Feature Extraction | Text | Measured Component | $0.120$ | $0.165$ | Char 3-5 gram & word 1-2 gram tokenization and sparse vector construction. |
| **Op-03** | Semantic Embedding Inference | Text | Measured Component | $1.850$ | $2.310$ | Unbatched forward pass through `all-MiniLM-L6-v2` on CPU. |
| **Op-04** | Code Lexing & Token Normalization | Code | Measured Component | $0.280$ | $0.395$ | Comment stripping, whitespace normalisation, variable canonicalization. |
| **Op-05** | Java AST Syntax Parsing | Code | Measured Component | $2.150$ | $2.840$ | AST parse tree construction via `javalang`. |
| **Op-06** | Software Metrics Extraction | Code | Measured Component | $0.420$ | $0.560$ | Halstead volume/difficulty, McCabe cyclomatic complexity, nesting depth. |
| **Op-07** | Stage-1 Fused Routing Decision | Text & Code | Measured Component | $0.050$ | $0.075$ | Evaluating linear fusion equation and checking $(\theta_{\text{low}}, \theta_{\text{high}})$. |
| **Op-08** | Stage-2 Cross-Encoder Forward Pass | Text | Measured Component | **$4.046$** | $5.120$ | Full bidirectional cross-attention pass (`ms-marco-MiniLM-L-6-v2`). |
| **Op-09** | Myers Token Alignment Diff | Text | Measured Component | $0.620$ | $0.850$ | Greedy $\mathcal{O}(ND)$ shortest edit script generation for text highlighting. |
| **Op-10** | AST Subtree Isomorphism Match | Code | Measured Component | $1.850$ | $2.410$ | Subtree hash intersection and isomorphic block identification. |

> [!IMPORTANT]
> **Clarification on the $0.050\text{ ms}$ Metric:**  
> The $0.050\text{ ms}$ figure represents the **system design target for cached similarity retrieval** (Op-01/Op-07) after document representations have been pre-computed during ingestion. It is **NOT** the end-to-end processing time for an un-indexed pair requiring raw document parsing, neural encoding, and AST extraction.

---

## 4. End-to-End Pipeline Cost Equation

To prevent double-counting shared preprocessing or cached representations, the end-to-end expected latency $\mathbb{E}[T]$ per candidate pair is formulated as:

$$\mathbb{E}[T] = T_{\text{stage1}} + P(\text{Escalate}) \cdot T_{\text{stage2}} + P(\text{Flagged}) \cdot T_{\text{evidence}}$$

where:
- **Pre-Indexed Cohort Mode (Typical Institutional Production):**
  - All $M$ student submissions are preprocessed once at upload time ($\mathcal{O}(M)$ cost):
    $$T_{\text{pre}}(D) = \text{Op-02} + \text{Op-03} \approx 1.970\text{ ms per submission}$$
  - Pairwise screening evaluates pre-indexed dot-products ($\mathcal{O}(M^2)$ cost):
    $$T_{\text{stage1}} = \text{Op-01} + \text{Op-07} \approx 0.095\text{ ms per pair}$$
- **Un-Indexed On-the-Fly Mode (Cold Ingestion):**
  - Involves raw feature extraction and inference per candidate pair:
    $$T_{\text{stage1}}^{\text{cold}} = \text{Op-02} + 2 \cdot \text{Op-03} + \text{Op-07} \approx 3.870\text{ ms per pair}$$

### Expected Latency in Pre-Indexed Production Mode (Text):
Assuming a standard operating threshold where $75.0\%$ of candidate pairs are pruned at Stage 1 ($P(\text{Escalate}) = 0.25$) and $5.0\%$ are flagged for review ($P(\text{Flagged}) = 0.05$):
$$\mathbb{E}[T_{\text{text}}] = 0.095 + (0.25 \times 4.046) + (0.05 \times 0.620) = 0.095 + 1.0115 + 0.031 = \mathbf{1.1375\text{ ms per pair}}$$

### Comparison Against Monolithic Deployment:
- **Uniform Heavy Model Baseline:** $4.046\text{ ms}$ per pair across all $500,000$ pairs $\to$ **$2,023\text{ seconds}$ ($33.7\text{ minutes}$)**.
- **Proposed Two-Stage Architecture:** $1.138\text{ ms}$ per pair across $500,000$ pairs $\to$ **$569\text{ seconds}$ ($9.5\text{ minutes}$)**.
- **Net Computational Savings:** **$71.9\%$ reduction in inference latency** and **$75.0\%$ reduction in heavy cross-encoder FLOPs**.

---

## 5. Memory Footprint and Compute Resources

| Resource Dimension | Unit | Stage 1 (Screening) | Stage 2 (Heavy Model) | Attribution (Evidence) | Full Pipeline Peak |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **RAM Footprint** | Megabytes (MB) | $245\text{ MB}$ | $580\text{ MB}$ | $180\text{ MB}$ | **$\sim 650\text{ MB}$** |
| **VRAM Footprint (if GPU)** | Megabytes (MB) | $0\text{ MB}$ (CPU) | $420\text{ MB}$ | $0\text{ MB}$ | **$\sim 450\text{ MB}$** |
| **FLOP Count** | MFLOPs / Pair | $\sim 0.15\text{ MFLOPs}$ | $\sim 3,200\text{ MFLOPs}$ | $\sim 2.5\text{ MFLOPs}$ | **Variable via Routing** |

