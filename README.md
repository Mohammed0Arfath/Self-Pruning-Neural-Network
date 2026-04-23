# Self-Pruning Neural Network on CIFAR-10

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
- The prunable layers introduce learnable gates that dynamically suppress unimportant connections during training.

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

---

## Results

| Lambda | Accuracy | Sparsity |
|--------|---------|----------|
| 1e-6   | 76.11%  | 34.68%   |
| 5e-6   | 75.47%  | 43.66%   |
| 1e-5   | 73.25%  | 45.23%   |

The best-performing model (λ = 1e-6) achieves 76.11% accuracy while pruning 34.68% of parameters.

<img width="669" height="568" alt="image" src="https://github.com/user-attachments/assets/093d3b48-c4a2-461b-a1e1-4efd81d74467" />

The trade-off curve shows a gradual decline in accuracy as sparsity increases, indicating that moderate pruning does not significantly harm performance.


---

## Analysis

- Increasing λ leads to higher sparsity but reduced accuracy  
- Moderate values of λ provide the best trade-off  
- The model removes approximately 40–45% of connections with minimal accuracy loss  
- Beyond a certain point, sparsity gains plateau while accuracy degrades more significantly  

---

## Gate Behavior

The distribution of gate values shows:

- A concentration near zero, representing pruned connections  
- A smaller subset of gates remaining active, representing important weights  

This distribution indicates effective sparsification:
- A dominant spike near zero shows a large number of connections have been suppressed
- A secondary distribution away from zero corresponds to important retained weights

This bimodal pattern confirms that the model learns structured sparsity rather than uniformly shrinking weights.

<img width="752" height="565" alt="image" src="https://github.com/user-attachments/assets/4a19dc5c-6c5f-42b6-83ed-0cd10914374f" />


---

## Key Insights

- Dense layers contain a significant number of redundant parameters  
- Sparsity can be introduced during training without post-processing  
- Learned pruning is more flexible than static pruning methods  

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
3. Install required dependencies (if needed):

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
└── README.md

```
---

## Future Work

- Extend pruning to convolutional layers  
- Explore L0 regularization for sharper sparsity  
- Evaluate inference speed improvements after pruning  
- Compare with magnitude-based pruning methods  

---
