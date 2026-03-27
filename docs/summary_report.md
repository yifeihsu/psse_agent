# PSSE Agent — Checkpoint Summary Report

**Date:** 2026-03-27
**HPC Cluster:** NYU Greene / `gh015.hpc.nyu.edu`
**GPU:** NVIDIA H100 80GB HBM3
**Base Model:** `unsloth/gpt-oss-20b-unsloth-bnb-4bit` (20B MoE, 4-bit NF4)
**Fine-tuning Method:** LoRA (rank-64, α-64) via Unsloth SFT
**W&B Run:** `psse-agent-sft/n7z7m9rv` (offline, 2026-03-20)

---

## 1. Dataset Overview

| Split | Samples |
|-------|---------|
| Train | 2,003   |
| Valid | 265     |
| Test  | 232     |
| **Total** | **2,500** |

**Scenario distribution (balanced, 500 per class):**

| Scenario | Count |
|---|---|
| `no_error` | 500 |
| `measurement_error` | 500 |
| `parameter_error` | 500 |
| `topology_error` | 500 |
| `harmonic_anomaly` | 500 |

Power system: **IEEE 14-bus** (14 buses, 20 branches).
Measurement vector: 122 channels — `[Vm, Pinj, Qinj, Pf, Qf, Pt, Qt]`.
Measurement noise: σ_Vm = 0.001 p.u., σ_inj = σ_flow = 0.01 p.u.

---

## 2. Training Configuration

| Hyperparameter | Value |
|---|---|
| LoRA rank / alpha | 64 / 64 |
| Target modules | `q/k/v/o_proj`, `gate/up/down_proj` |
| Max sequence length | 16,384 tokens |
| Batch size (per device) | 4 |
| Gradient accumulation | 4 (effective batch = 16) |
| Learning rate | 2e-4 (cosine schedule) |
| Warmup steps | 20 |
| Epochs | 1 |
| Optimizer | AdamW-8bit |
| Precision | bfloat16 |
| Gradient checkpointing | Yes |

**Total training time:** ~29 min (1,748 s)
**Throughput:** 0.801 samples/s, 0.05 steps/s
**Total FLOPs:** ~1.28 × 10¹⁸

---

## 3. Training Loss Curve

| Step | Epoch | Train Loss | Grad Norm | LR |
|------|-------|-----------|-----------|-----|
| 5 | 0.11 | 0.5592 | 2.121 | 9.0e-5 |
| 10 | 0.17 | 0.2486 | 1.114 | 1.4e-4 |
| 15 | 0.23 | 0.0877 | 1.457 | 1.9e-4 |
| 20 | 0.29 | 0.0416 | 1.334 | 1.98e-4 |
| 25 | 0.34 | 0.0395 | 1.390 | 1.92e-4 |
| 30 | 0.40 | 0.0269 | 0.758 | 1.80e-4 |
| 35 | 0.46 | 0.0239 | 0.825 | 1.64e-4 |
| 40 | 0.51 | 0.0238 | 1.026 | 1.45e-4 |
| 45 | 0.57 | 0.0179 | 0.324 | 1.23e-4 |
| 50 | 0.63 | 0.0270 | 0.694 | 1.00e-4 |
| 55 | 0.69 | 0.0196 | 0.300 | 7.71e-5 |
| 60 | 0.74 | 0.0136 | 0.429 | 5.54e-5 |
| 65 | 0.80 | 0.0134 | 0.364 | 3.61e-5 |
| 70 | 0.86 | 0.0105 | 0.244 | 2.02e-5 |
| 75 | 0.91 | 0.0172 | 0.571 | 8.52e-6 |
| 80 | 0.97 | 0.0122 | 0.381 | 1.70e-6 |
| **Final** | **1.00** | **0.0122** | **0.381** | **1.70e-6** |

**Final train loss: 0.1690** (epoch-averaged), **0.0122** (last step).

---

## 4. Evaluation Metrics — Test Set (n = 232)

### 4.1 Error-Family Identification Accuracy

| Class | Correct | Total | Accuracy |
|---|---|---|---|
| `no_error` | 42 | 42 | **100.0%** |
| `measurement_error` | 48 | 48 | **100.0%** |
| `parameter_error` | 43 | 43 | **100.0%** |
| `topology_error` | 51 | 51 | **100.0%** |
| `harmonic_anomaly` | 48 | 48 | **100.0%** |
| **Overall** | **232** | **232** | **100.0%** |

### 4.2 Confusion Matrix (rows = ground truth, cols = predicted)

|  | no_error | meas. | param. | topo. | harmonic |
|---|---|---|---|---|---|
| `no_error` | **42** | 0 | 0 | 0 | 0 |
| `measurement_error` | 0 | **48** | 0 | 0 | 0 |
| `parameter_error` | 0 | 0 | **43** | 0 | 0 |
| `topology_error` | 0 | 0 | 0 | **51** | 0 |
| `harmonic_anomaly` | 0 | 0 | 0 | 0 | **48** |

Zero off-diagonal entries — the agent perfectly separates all five error families on the test split.

### 4.3 Confidence

| Metric | Value |
|---|---|
| Mean confidence | **0.9799** |
| Min confidence | 0.950 |
| Max confidence | 0.990 |

---

## 5. Tool-Calling Analysis — Test Set

### 5.1 Tool Call Frequency

| Tool | Calls | Share |
|---|---|---|
| `wls_from_path` *(WLS state estimation)* | 331 | 63.5% |
| `correct_topology_from_path` | 51 | 9.8% |
| `correct_measurements_from_path` | 48 | 9.2% |
| `run_hse_from_path` *(harmonic SE)* | 48 | 9.2% |
| `correct_parameters_from_path` | 43 | 8.3% |
| **Total** | **521** | 100% |

### 5.2 Calls per Trace

| Stat | Value |
|---|---|
| Mean | 2.25 |
| Min | 1 |
| Max | 3 |

All traces begin with `wls_from_path` as mandated by the decision policy. The agent then dispatches exactly one correction or harmonic-SE tool when an anomaly is detected, and re-runs WLS to confirm.

---

## 6. Sample Outputs

### no_error — Sample #1
- **Tools:** `wls_from_path`
- **Verdict:** `error_family: ["no_error"]`, `has_error: false`, confidence: 0.98
- **Correct:** ✓

### measurement_error — Sample #43
- **Tools:** `wls_from_path` → `correct_measurements_from_path` → `wls_from_path`
- **Verdict:** `error_family: ["measurement_error"]`, `has_error: true`, confidence: 0.99
- **Correct:** ✓
- *Concentrated large normalized residuals triggered the measurement-correction branch.*

### parameter_error — Sample #91
- **Tools:** `wls_from_path` → `correct_parameters_from_path`
- **Verdict:** `error_family: ["parameter_error"]`, `has_error: true`, confidence: 0.99
- **Correct:** ✓
- *Large Lagrange multipliers concentrated on one branch identified the faulty line parameters.*

### topology_error — Sample #134
- **Tools:** `wls_from_path` → `correct_topology_from_path` → `wls_from_path`
- **Verdict:** `error_family: ["topology_error"]`, `has_error: true`, confidence: 0.99
- **Correct:** ✓
- *Widespread residual patterns with no dominant measurement pointed to topology mismatch.*

### harmonic_anomaly — Sample #185
- **Tools:** `wls_from_path` → `run_hse_from_path`
- **Verdict:** `error_family: ["harmonic_anomaly"]`, `has_error: true`, confidence: 0.95
- **Correct:** ✓
- *Elevated global residual without a dominant bad measurement triggered HSE.*

---

## 7. Model Checkpoint

| Item | Detail |
|---|---|
| Output dir | `/scratch/hk4488/psse_agent/outputs/gpt_oss_sft` |
| Save steps | Every 100 steps, keep last 2 |
| Total steps | 88 |
| Checkpoints saved | `checkpoint-88` (final) |
| Model parameters | 20.95B (base), LoRA adds ~160M trainable |

---

## 8. Summary

The SFT run successfully fine-tuned a 20B MoE model (`gpt-oss-20b`) on the balanced PSSE diagnostic dataset. Training loss converged from **0.56 → 0.012** over one epoch (~29 min on one H100). The resulting agent achieves **100% identification accuracy** on the 232-sample test set across all five error families, with a mean decision confidence of **0.98**. The agentic tool-call pattern is consistent with the specified decision policy: every trace begins with `wls_from_path`, and the subsequent correction tool is chosen correctly based on residual/Lagrange diagnostics.
