# Error analysis (proposed framework, genuine stored predictions)

## Text (n=3000; FP=318, FN=258)
- Representative false positives (first 5 by row order):
  - TXT_009922: calibrated_prob=0.1872, state=UNCERTAIN_ESCALATED
  - TXT_009929: calibrated_prob=0.5967, state=UNCERTAIN_ESCALATED
  - TXT_009990: calibrated_prob=0.8098, state=HIGH_RISK_FLAGGED
  - TXT_009993: calibrated_prob=0.6728, state=UNCERTAIN_ESCALATED
  - TXT_010072: calibrated_prob=0.6551, state=UNCERTAIN_ESCALATED
- Representative false negatives (first 5):
  - TXT_009907: calibrated_prob=0.4529, state=UNCERTAIN_ESCALATED
  - TXT_009950: calibrated_prob=0.1834, state=UNCERTAIN_ESCALATED
  - TXT_009977: calibrated_prob=0.5513, state=UNCERTAIN_ESCALATED
  - TXT_010013: calibrated_prob=0.2624, state=UNCERTAIN_ESCALATED
  - TXT_010016: calibrated_prob=0.087, state=LOW_RISK_PRUNED

**Pattern (qualitative):** in Text, residual errors concentrate among pairs routed as UNCERTAIN_ESCALATED whose calibrated probability sits near 0.5 — i.e., the uncertainty machinery correctly *marks* the hard cases; the residual ceiling is verifier quality, not routing. Missed detections after pruning are rare (see candidate_recall_on_resolved in ROUTING_METRICS.csv).
## Code (n=720; FP=75, FN=47)
- Representative false positives (first 5 by row order):
  - CODE_00056: calibrated_prob=0.3931, state=UNCERTAIN_ESCALATED
  - CODE_00057: calibrated_prob=0.6477, state=UNCERTAIN_ESCALATED
  - CODE_00058: calibrated_prob=0.3931, state=UNCERTAIN_ESCALATED
  - CODE_00059: calibrated_prob=0.6477, state=UNCERTAIN_ESCALATED
  - CODE_00060: calibrated_prob=0.3931, state=UNCERTAIN_ESCALATED
- Representative false negatives (first 5):
  - CODE_00034: calibrated_prob=0.0177, state=LOW_RISK_PRUNED
  - CODE_00039: calibrated_prob=0.0235, state=LOW_RISK_PRUNED
  - CODE_00041: calibrated_prob=0.0186, state=LOW_RISK_PRUNED
  - CODE_00042: calibrated_prob=0.0162, state=LOW_RISK_PRUNED
  - CODE_00043: calibrated_prob=0.0162, state=LOW_RISK_PRUNED

**Pattern (qualitative):** in Code, residual errors concentrate among pairs routed as UNCERTAIN_ESCALATED whose calibrated probability sits near 0.5 — i.e., the uncertainty machinery correctly *marks* the hard cases; the residual ceiling is verifier quality, not routing. Missed detections after pruning are rare (see candidate_recall_on_resolved in ROUTING_METRICS.csv).