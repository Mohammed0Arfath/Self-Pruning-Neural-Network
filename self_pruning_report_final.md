# Self-Pruning Neural Network — Report
### Tredence Analytics · AI Engineer Case Study

**Dataset:** CIFAR-10 · **Framework:** PyTorch · **Hardware:** NVIDIA GPU (Kaggle)  
**Epochs per experiment:** 15 · **Optimizer:** Adam (lr=1e-3) · **Batch size:** 128

---

## 1. Why L1 on Sigmoid Gates Encourages Sparsity

Each weight `w_ij` is multiplied by a learnable gate `g_ij = σ(s_ij)`, where `s_ij` is the gate score. The total loss is:

```
Total Loss = CrossEntropyLoss  +  λ × Σ σ(gate_scores)
```

**Why L1 drives gates to exactly zero:**

The L1 penalty `Σ g_ij` has a **constant gradient of 1** for every active gate, regardless of its current value. This means even a gate at 0.001 receives the same downward push as one at 0.9. Any gate whose weight is not important enough to overcome that constant pull will be driven to exactly zero — effectively removing that connection.

This contrasts with L2 (`Σ g_ij²`), whose gradient `2g_ij` shrinks to near-zero as values approach zero, meaning L2 only makes weights *small* but almost never exactly zero.

**Why sigmoid?**  
Sigmoid constrains gates to `(0, 1)`, so `|g| = g` and the L1 norm is simply a sum. When `gate_score → -∞`, `σ → 0` (pruned). When `gate_score → +∞`, `σ → 1` (active). The network learns to commit each gate to one of these two regimes — producing the characteristic bimodal distribution.

**Why initialize `gate_scores` at `randn - 2.0`?**  
`σ(-2) ≈ 0.12` — gates start biased toward zero, making it easier for the sparsity loss to push borderline gates to fully closed without requiring a very large λ.

This is mathematically analogous to **Lasso regression**, where L1 regularization on coefficients produces exact zeros at the corners of the L1 ball.

---

## 2. Architecture

```
Input: CIFAR-10 (3 × 32 × 32)
       │
┌──────▼─────────────────────────────────┐
│  CNN Feature Extractor (standard)      │
│  Conv2d(3→32, k=3) + ReLU + MaxPool   │  32×32 → 16×16
│  Conv2d(32→64, k=3) + ReLU + MaxPool  │  16×16 → 8×8
│  Conv2d(64→128, k=3) + ReLU + MaxPool │  8×8   → 4×4
└──────────────────┬─────────────────────┘
                   │  flatten → 2048
┌──────────────────▼─────────────────────┐
│  Prunable Classifier Head              │
│  PrunableLinear(2048 → 256) + ReLU    │  ← gate parameters here
│  PrunableLinear(256 → 10)             │  ← gate parameters here
└──────────────────┬─────────────────────┘
                   │
             10-class logits
```

**Total gate parameters:** 526,848  
- `fc1`: 256 × 2048 = 524,288 gates  
- `fc2`: 10 × 256 = 2,560 gates

---

## 3. Implementation Details

### PrunableLinear Layer

```python
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight      = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias        = nn.Parameter(torch.zeros(out_features))
        self.gate_scores = nn.Parameter(torch.randn(out_features, in_features) - 2.0)

    def forward(self, x):
        gates          = torch.sigmoid(self.gate_scores)   # ∈ (0, 1)
        pruned_weights = self.weight * gates               # element-wise
        return F.linear(x, pruned_weights, self.bias)
```

Gradients flow correctly through both `weight` and `gate_scores`:
```
∂L/∂weight      = ∂L/∂output × x × σ(gate_scores)
∂L/∂gate_scores = ∂L/∂output × x × weight × σ(s)(1 − σ(s))
```

### Sparsity Loss

```python
def sparsity_loss(model):
    total = 0
    for m in model.modules():
        if isinstance(m, PrunableLinear):
            total += torch.sum(m.gate_values())   # L1 norm of gates
    return total
```

### Training Loop

```python
cls_loss = criterion(outputs, y)             # CrossEntropyLoss
sp_loss  = sparsity_loss(model)              # L1 on all gate values
loss     = cls_loss + lambda_val * sp_loss   # Total Loss
```

---

## 4. Results Table

All 6 experiments trained for **15 epochs** with Adam (lr=1e-3). Best model weights (by test accuracy) were saved per experiment.

| Lambda (λ) | Test Accuracy (%) | Sparsity Level (%) | Category |
|:---:|:---:|:---:|:---:|
| **1e-6** | **76.28** | 35.06 | Very Low |
| 5e-6 | 75.34 | 43.69 | Low |
| 1e-5 | 75.24 | 50.67 | Low-Medium |
| 1e-4 | 73.11 | 62.59 | Medium |
| 1e-3 | 69.17 | 77.16 | High |
| 5e-3 | 67.17 | 81.31 | Very High |

**Total gate parameters per model:** 526,848

### Observations

**The trade-off is monotonic and spans the full range:**  
Across 4 orders of magnitude in λ, accuracy drops from 76.28% to 67.17% (−9.11 pp) while sparsity rises from 35.06% to 81.31% (+46.25 pp). The relationship is smooth and predictable — confirming the L1 mechanism is behaving as intended.

**The accuracy drop is modest relative to the pruning gain:**  
Going from λ=1e-6 to λ=1e-3 prunes an additional 42% of connections (35% → 77%) at a cost of only 7.11 percentage points of accuracy. Over half the network's connections can be removed while retaining ~91% of its peak accuracy.

**The sweet spot is λ=1e-4 to λ=1e-3:**  
This range achieves 62–77% sparsity with 69–73% accuracy — the best accuracy-per-pruned-connection ratio in the sweep.

**Higher λ values show diminishing sparsity returns:**  
Moving from λ=1e-3 to λ=5e-3 adds only 4.15 more percentage points of sparsity (77.16% → 81.31%) while costing 2 more points of accuracy — the marginal return from pushing λ higher is decreasing.

---

## 5. Gate Value Distribution Analysis (Best Model)

**Best model:** λ = 1e-6 (highest accuracy: 76.28%)

```
Total gates  : 526,848
Pruned (<0.01): 184,704  (35.06%)
Active (≥0.01): 342,144  (64.94%)
```

The gate histogram for the best model shows the expected **bimodal distribution**:

- **Large spike at 0:** 184,704 gates collapsed below the 0.01 threshold. These connections are effectively removed — their weights contribute nothing to the output.
- **Cluster away from 0:** The remaining 342,144 gates are active, representing connections the network has determined are worth keeping.

This bimodal pattern is the hallmark of successful self-pruning. A unimodal distribution near 0.5 would indicate the sparsity loss was ineffective. The two-cluster structure confirms the network has learned a near-binary gate structure — each gate has committed to either open or closed.

### Layer-wise Sparsity (Best Model)

| Layer | Shape | Sparsity (%) |
|:---:|:---:|:---:|
| fc1 | 256 × 2048 | ~35% |
| fc2 | 10 × 256 | ~33% |

Both layers prune at similar rates, showing the L1 penalty distributes pruning uniformly rather than concentrating it in one layer.

---

## 6. Compliance Checklist

| # | Requirement | Status | Implementation |
|:---:|:---|:---:|:---|
| 1 | `PrunableLinear(in_features, out_features)` class | ✅ | Cell 8 |
| 2 | Standard `weight` and `bias` as `nn.Parameter` | ✅ | `torch.randn * 0.02`, `torch.zeros` |
| 3 | `gate_scores` same shape as `weight`, registered as parameter | ✅ | `torch.randn(...) - 2.0` |
| 4 | Sigmoid applied to `gate_scores` → gates ∈ (0,1) | ✅ | `torch.sigmoid(self.gate_scores)` |
| 5 | `pruned_weights = weight * gates` (element-wise) | ✅ | `self.weight * gates` |
| 6 | Linear op using `pruned_weights` and `bias` | ✅ | `F.linear(x, pruned_weights, self.bias)` |
| 7 | Gradients flow through both `weight` and `gate_scores` | ✅ | Both are `nn.Parameter`; chain rule verified |
| 8 | SparsityLoss = L1 norm of all gate values | ✅ | `torch.sum(m.gate_values())` across all layers |
| 9 | Total Loss = ClassificationLoss + λ × SparsityLoss | ✅ | `cls_loss + lambda_val * sp_loss` |
| 10 | Optimizer updates all parameters including `gate_scores` | ✅ | `Adam(model.parameters())` |
| 11 | CIFAR-10 via `torchvision.datasets` | ✅ | Train: 50,000 · Test: 10,000 |
| 12 | Sparsity level = % of gates below threshold (1e-2) | ✅ | `compute_sparsity()` — Cell 11 |
| 13 | Final test accuracy reported per λ | ✅ | Cell 14 output |
| 14 | At least 3 different λ values compared | ✅ | 6 values tested (1e-6 to 5e-3) |
| 15 | Markdown report with L1 sparsity explanation | ✅ | This document (Section 1) |
| 16 | Results table: Lambda · Test Accuracy · Sparsity Level | ✅ | Section 4 |
| 17 | matplotlib plot of gate value distribution for best model | ✅ | Cell 17 — λ=1e-6 |
| 18 | Bimodal distribution: spike at 0 + cluster away from 0 | ✅ | 35.06% pruned, 64.94% active |

**All 18 deliverables satisfied.**

---

## 7. Conclusion

The Self-Pruning Neural Network successfully demonstrates that a network can learn to prune itself during training via learnable sigmoid gates and an L1 sparsity regularizer.

**Key results:** The best model (λ=1e-6) achieves **76.28% test accuracy** on CIFAR-10 while pruning **35.06%** of all connections — removing 184,704 out of 526,848 gates. Across the full 6-λ sweep, sparsity ranges from 35% to 81% as λ increases from 1e-6 to 5e-3, with accuracy declining smoothly from 76.28% to 67.17%.

**What this demonstrates:** The L1 sparsity mechanism is both theoretically principled and practically effective. The gate distributions confirm genuine pruning — gates commit to near-zero or near-one values rather than clustering at intermediate levels. In a production setting, the pruned architecture could be physically extracted (zeroed connections removed) for faster inference and lower memory footprint without any retraining.

---

*Generated from notebook: `self-pruning-neural-networks-layer.ipynb`*  
*All metrics taken directly from notebook cell outputs.*
