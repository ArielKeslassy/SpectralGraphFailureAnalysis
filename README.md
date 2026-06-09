# Spectral Graph Failure Analysis for DiT Hallucination Detection

This repository implements a research framework for detecting hallucinations in Diffusion Transformers (DiT) by analyzing the spectral properties of their internal attention mechanisms. The core hypothesis is that structural anomalies in the attention graphs—captured via spectral graph theory—correlate with generation failures (hallucinations).

## 1. Data Generation (`data.py`)

The data generation process involves running inference on a diffusion model while capturing its internal state.

*   **Model**: Uses `stabilityai/sdxl-turbo` via the Hugging Face `diffusers` library.
*   **Attention Extraction**:
    *   A custom attention processor (`AttnProcessorWithWeights`) is registered to the U-Net model.
    *   It specifically targets cross-attention layers (modules ending in `attn2`).
    *   During the forward pass, it intercepts the attention scores (softmaxed $QK^T$), detaches them from the computation graph, and moves them to CPU storage to save memory.
*   **Output**:
    *   **Images**: Generated images are saved as `.png` files.
    *   **Attention Maps**: Per-image attention weights are saved as `.pt` (PyTorch) files. These contain the raw attention tensors for specific layers/heads.
*   **Annotations**: Ground truth labels (indicating hallucinations or specific failure modes) are loaded from a CVAT-format XML file (`annotations.xml`).

## 2. Data Processing & Feature Extraction

The raw attention maps are transformed into graph representations, from which spectral and graph-theoretic features are extracted.

### A. Graph Construction (`graph_utils.py`)
Each attention matrix is treated as the adjacency matrix of a **bipartite graph** connecting query tokens (spatial locations) to key tokens (text embeddings).

1.  **Sparsification**: To reduce noise and computational load, the attention matrix is sparsified by zeroing out values below a certain quantile (`sparse_factor`).
2.  **Laplacian Computation**:
    *   Computes the **Normalized Laplacian** matrix ($L = I - D^{-1/2} A D^{-1/2}$) or unnormalized Laplacian.
    *   This step is optimized to run on GPU/MPS using sparse matrix operations.

### B. Spectral Decomposition
The system computes the **spectral embedding** of the attention graph:
*   **Eigensystem Solver**: Uses Locally Optimal Block Preconditioned Conjugate Gradient (LOBPCG) or standard eigensolvers to find the $k$ smallest eigenvalues and eigenvectors of the Laplacian.
*   **Optimization**: The solver is batched and hardware-accelerated (CUDA/MPS) to handle large numbers of attention heads efficiently.

### C. Feature Engineering (`spectral_graph_features.py`)
A rich set of features is derived from the spectral decomposition and the graph structure.

#### 1. Spectral Statistics
Derived directly from the eigenvalues ($\lambda_0 \le \lambda_1 \le \dots \le \lambda_k$):
*   **Algebraic Connectivity**: The second smallest eigenvalue ($\lambda_1$), representing how well-connected the graph is.
*   **Spectral Gap**: Difference between consecutive eigenvalues (e.g., $\lambda_1 - \lambda_0$).
*   **Spectral Entropy**: Shannon entropy of the normalized eigenvalues, measuring the complexity of the graph structure.
*   **Eigenvalue Distribution**: Mean, standard deviation, and raw values of the top-$k$ eigenvalues.

#### 2. Graph-Theoretic Metrics
Derived from eigenvectors and graph topology (with approximations for speed):
*   **Fiedler Vector Variance**: Variance of the eigenvector corresponding to $\lambda_1$, indicating partitioning stability.
*   **Eigenvector Centrality**: Measure of node influence based on the principal eigenvector.
*   **Betweenness Centrality**: Quantifies how often a node acts as a bridge along the shortest path between two other nodes.
    *   *Optimization*: Uses random sampling approximation for large graphs (5-20x speedup).
*   **Closeness Centrality**: Average length of the shortest path between the node and all other nodes.
    *   *Optimization*: Uses a path length cutoff approximation.
*   **Degree Centrality & Freeman Centrality**: Measures of immediate connectivity and centralization.

#### 3. Aggregation
Since a single image generation involves multiple attention heads across layers, features are aggregated (Mean, Max, Std) to produce a single feature vector per image.

## 3. Evaluation & Metrics (`probes.py`, `metrics.py`)

The extracted features are evaluated on their ability to distinguish between "clean" and "hallucinated" generations using linear probes.

### A. Probing Methodology
*   **Model**: Logistic Regression with balanced class weights.
*   **Validation Strategy**: Nested Stratified K-Fold Cross-Validation.
    *   **Outer Loop**: Splits data into Train/Test sets ($k=5$).
    *   **Inner Loop**: Splits Train data to tune the decision threshold.
*   **Normalization**: Features are standardized (Z-score normalization) within each fold to prevent data leakage.

### B. Threshold Tuning
The decision threshold for the Logistic Regression classifier is not fixed at 0.5. Instead, it is tuned on the inner validation set to maximize the **Matthews Correlation Coefficient (MCC)**, ensuring robustness against class imbalance.

### C. Performance Metrics
The following metrics are reported for each feature set and label:
*   **ROC-AUC**: Area Under the Receiver Operating Characteristic curve.
*   **PR-AUC**: Area Under the Precision-Recall curve.
*   **MCC**: Matthews Correlation Coefficient (primary metric for imbalanced binary classification).
*   **F1 Score**: Harmonic mean of precision and recall.
*   **Balanced Accuracy**: Arithmetic mean of sensitivity and specificity.

### D. Experiment Tracking
All metrics, along with feature configurations, are logged to **Weights & Biases (wandb)** for experiment tracking and visualization.
