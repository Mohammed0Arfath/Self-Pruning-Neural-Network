
# Self-Pruning Neural Network on CIFAR-10
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-blue)](https://www.kaggle.com/code/mohammedarfathr/self-pruning-neural-networks-layer)

## Overview

This project implements a neural network that learns to prune its own parameters during training. Instead of manually removing weights after training, the model learns which connections are unnecessary and suppresses them through a differentiable gating mechanism.

The objective is to achieve an optimal balance between model accuracy and sparsity, demonstrating that a significant portion of parameters in dense layers can be removed without substantial performance degradation.

---

## Methodology

### Prunable Linear Layer

A custom linear layer is implemented where each weight is associated with a learnable gate parameter.

For a given weight matrix `W`, a corresponding gate score matrix `S` is introduced. The effective weights are computed as:

```
W' = W * sigmoid(S)
```

Where:
- `sigmoid` ensures gate values lie between 0 and 1  
- Gate values control the importance of each connection  

This allows the model to continuously learn which connections are useful.

---

### Sparsity Regularization

The loss function is defined as:

```
Total Loss = CrossEntropyLoss + λ * sum(gates)
```

Where:
- CrossEntropyLoss ensures classification performance  
- The second term is the L1 norm of gate values  
- `λ` controls the trade-off between accuracy and sparsity  

This encourages the network to drive unnecessary gates toward zero.

---

## Architecture

The model uses a hybrid architecture:

- Convolutional layers for feature extraction  
- Custom prunable linear layers for classification and sparsity learning  
- The prunable layers introduce learnable gates that dynamically suppress unimportant connections during training  

```
Input (32×32×3)
↓
Convolutional Layers
↓
Feature Flattening
↓
Prunable Linear Layer
↓
Prunable Linear Layer
↓
Output (10 classes)
```

![Architecture](https://github.com/user-attachments/assets/d8b7ad50-8035-40db-8365-104731ded272)
*Figure: Model architecture showing convolutional backbone and prunable layers.*

---

## Experimental Setup

- Dataset: CIFAR-10  
- Optimizer: Adam  
- Loss: CrossEntropy + λ × Sparsity Loss  

Evaluation Metrics:
- Test Accuracy  
- Sparsity Level (percentage of gates below threshold)  

### Sparsity Definition

Sparsity is defined as the percentage of connections whose gate values fall below a threshold (1e-2):

```
Sparsity (%) = (Number of gates < 0.01 / Total gates) × 100
```

This threshold identifies connections that contribute negligibly to the output.

---

## Results

| Lambda | Accuracy (%) | Sparsity (%) |
|--------|-------------|--------------|
| 1e-6   | 76.28       | 35.06        |
| 5e-6   | 75.34       | 43.69        |
| 1e-5   | 75.24       | 50.67        |
| 1e-4   | 73.11       | 62.59        |
| 1e-3   | 69.17       | 77.16        |
| 5e-3   | 67.17       | 81.31        |

These results demonstrate a clear and consistent sparsity–accuracy trade-off across a wide range of λ values.

### Key Observations

- Increasing λ steadily increases sparsity, confirming that the regularization term effectively suppresses connections  
- Accuracy degrades gradually rather than abruptly, indicating that a large portion of parameters are redundant  
- The model remains relatively stable up to ~50–60% sparsity before sharper accuracy degradation  

### Best Trade-off

- **λ = 1e-6 → Highest accuracy (76.28%)**
- **λ = 1e-5 → Balanced trade-off (~50% sparsity with minimal accuracy drop)**

This shows that nearly **half of the network connections can be removed** with only a small reduction in performance.

![Tradeoff](https://github.com/user-attachments/assets/3d90df7e-c37c-4cee-b3cd-768b80ffe73a)
*Figure: Accuracy vs Sparsity trade-off across different λ values.*
---

## Analysis

- The sparsity–accuracy trade-off follows a smooth and predictable trend  
- Low λ values preserve accuracy but result in limited pruning  
- Moderate λ values (1e-5 to 1e-4) provide the best balance between efficiency and performance  
- High λ values (>1e-3) lead to aggressive pruning (>75%), but at a noticeable cost to accuracy  

A key observation is that:
- Up to ~50% sparsity, the model retains most of its predictive capability  
- Beyond ~60–70% sparsity, accuracy degradation becomes more pronounced  

This confirms that dense layers contain significant redundancy and can be compressed effectively using learned gating mechanisms.
---

## Gate Behavior

The distribution of gate values reveals how the model learns sparsity:

- A large spike near zero indicates that many connections are effectively suppressed  
- A smaller distribution away from zero represents important retained weights  

This behavior is a direct consequence of the L1 penalty:

- The L1 term encourages many gates to shrink toward zero  
- Important connections resist this penalty and remain active  

This results in a **bimodal distribution**, demonstrating structured sparsity rather than uniform shrinkage.

![Gates](https://github.com/user-attachments/assets/6f6739a0-a4d9-4e46-91c9-01c6cc16b307)
*Figure: Gate value distribution showing pruned vs active connections.*

---

## Key Insights

- Dense layers contain a significant number of redundant parameters  
- Sparsity can be introduced during training without post-processing  
- Learned pruning is more adaptive than static pruning techniques
- The model maintains strong performance even after removing over 50% of connections, highlighting substantial redundancy in dense layers  

---

## How to Run

This project is implemented as a notebook and can be run on Kaggle, Google Colab, or locally.

---

### Option 1: Run on Kaggle (Recommended)

1. Open the notebook on Kaggle  
2. Enable GPU (optional but recommended)  
3. Run all cells  

> The notebook includes dataset loading and all required dependencies.

---

### Option 2: Run on Google Colab

1. Upload the notebook (`self-pruning-neural-networks-layer.ipynb`) to Colab  
2. Enable GPU:
   - Runtime → Change runtime type → GPU  
3. Install dependencies (if needed):

```
pip install torch torchvision
```

4. Run all cells  

---

### Option 3: Run Locally

1. Download the notebook:
```
self-pruning-neural-networks-layer.ipynb
```

2. Open using Jupyter Notebook or VS Code:
```
jupyter notebook
```

3. Ensure the following dependencies are installed:

- Python 3.x  
- PyTorch  
- torchvision  
- NumPy  
- matplotlib  

---

## Repository Structure

```
.
├── self-pruning-neural-networks-layer.ipynb
├── README.md
└── Report.md
```

---

## Future Work

- Extend pruning to convolutional layers  
- Explore L0 regularization for sharper sparsity  
- Evaluate inference speed improvements after pruning  
- Compare with magnitude-based pruning methods  

---

## Author

Arfath  
AI/ML Engineering Student
