## Part 4: Confidence & Calibration

### 4.1 Calibration Curve

| Confidence Bin | Model Confidence | Actual Accuracy |
|----------------|------------------|-----------------|
| 0.3-0.4 | 32% | 100% |
| 0.5-0.6 | 56% | 100% |
| 0.7-0.8 | 74% | 75% |
| 0.8-0.9 | 84% | 86% |
| 0.9-1.0 | 94% | 100% |

**Expected Calibration Error (ECE)**: 0.167 (moderate)

### 4.2 Hallucination Detection

**AUC**: 0.52 (essentially random)

The model's output confidence is not sufficient to distinguish correct from incorrect facts. Better detection would require:
- Attention pattern analysis (sharpness metrics)
- Hidden state geometry
- Multi-layer probe ensembles

---
