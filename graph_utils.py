import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
import igraph as ig
from typing import Tuple, List, Optional


def build_bipartite_laplacian_and_eigensystem_sparse(
        attention_matrix: torch.Tensor,
        k_eigenvalues: int = 10,
        normalized: bool = True,
        sparse_factor: float = 0.1,
        compute_graph_features: bool = True,  # NEW: Allow skipping graph creation
        device: Optional[str] = None,  # NEW: Explicit device control
) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[ig.Graph]]]:
    """
    Builds sparse bipartite graph Laplacian and computes top-k smallest eigensystem using LOBPCG.

    OPTIMIZATIONS:
    - Batched degree/centrality computations on GPU/MPS
    - Automatic device detection (CUDA > MPS > CPU)
    - Delayed graph creation (only if needed)
    - Pre-computed centrality metrics stored as tensors
    - Minimized CPU transfers

    Args:
        attention_matrix: [batch_size, n_queries, n_keys] or [batch_size, n_heads, n_queries, n_keys]
        k_eigenvalues: Number of smallest eigenvalues to compute
        normalized: Use symmetric normalized Laplacian
        sparse_factor: Bottom quantile to zero out
        compute_graph_features: Whether to create igraph objects (set False to skip graph ops)
        device: Device to use ('cuda', 'mps', 'cpu', or None for auto-detect)

    Returns:
        eigenvalues: [batch_size, k_eigenvalues] or [batch_size, n_heads, k_eigenvalues]
        eigenvectors: [batch_size, n_total, k_eigenvalues] or [batch_size, n_heads, n_total, k_eigenvalues]
        graphs: List of igraph graphs with pre-computed metrics, or None if compute_graph_features=False
    """
    # Auto-detect and move to best available device
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

    # Move input to device if not already there
    if attention_matrix.device.type != device:
        attention_matrix = attention_matrix.to(device)

    reshape_output = False
    if attention_matrix.dim() == 4:
        b, h, q, k = attention_matrix.shape
        attention_matrix = attention_matrix.reshape(b * h, q, k)
        reshape_output = True
        original_shape = (b, h)
    elif attention_matrix.dim() != 3:
        raise ValueError(f"Unsupported attention matrix shape: {attention_matrix.shape}")

    b_size, n_q, n_k = attention_matrix.shape
    n_total = n_q + n_k

    # Input validation
    assert attention_matrix.min() >= -1e-6, "Attention weights should be non-negative"
    assert attention_matrix.max() <= 1 + 1e-6, "Attention weights should be <= 1"
    assert k_eigenvalues < n_total, f"k_eigenvalues ({k_eigenvalues}) must be < n_total ({n_total})"

    # OPTIMIZATION: Pre-allocate output tensors on correct device
    device_obj = attention_matrix.device
    dtype = attention_matrix.dtype
    
    # Use float32 for internal computations if input is low precision to avoid overflow/underflow
    # e.g. 1/sqrt(1e-12) = 1e6 which overflows float16 (max ~6.5e4)
    computation_dtype = torch.float32 if dtype in [torch.float16, torch.bfloat16] else dtype

    # OPTIMIZATION: Batch sparsification
    if sparse_factor > 0:
        q_val = torch.quantile(attention_matrix.float(), sparse_factor, dim=2, keepdim=True)
        mask = attention_matrix >= q_val
        attention_sparse = (attention_matrix * mask).to(dtype=computation_dtype)
    else:
        attention_sparse = attention_matrix.to(dtype=computation_dtype)

    # 1. Batched Normalized Out-Degree
    received_attention = torch.sum(attention_sparse, dim=1)  # [B, n_k]
    indices_k = torch.arange(n_k, device=device_obj)
    denominator = torch.clamp(n_q - indices_k, min=1.0)
    deg_k = received_attention / denominator # [B, n_k]
    deg_q = torch.ones(b_size, n_q, device=device_obj, dtype=computation_dtype)
    deg = torch.cat([deg_q, deg_k], dim=1) # [B, n_total]
    deg = torch.clamp(deg, min=1e-12)

    # 2. Batched Bipartite Adjacency Matrix
    A_full = torch.zeros(b_size, n_total, n_total, dtype=computation_dtype, device=device_obj)
    A_full[:, :n_q, n_q:] = attention_sparse
    A_full[:, n_q:, :n_q] = attention_sparse.transpose(1, 2)
    
    # 3. Batched Laplacian Construction
    if normalized:
        deg_inv_sqrt = torch.pow(deg, -0.5) # [B, n_total]
        # D^(-1/2) * A * D^(-1/2)
        # Using broadcasting: deg_inv_sqrt.unsqueeze(-1) * A_full * deg_inv_sqrt.unsqueeze(1)
        normalized_A = A_full * deg_inv_sqrt.unsqueeze(-1) * deg_inv_sqrt.unsqueeze(1)
        # L = I - normalized_A
        I = torch.eye(n_total, device=device_obj, dtype=computation_dtype).unsqueeze(0)
        laplacians = I - normalized_A
    else:
        D = torch.diag_embed(deg)
        laplacians = D - A_full

    # 4. Batched Eigensystem Computation
    # Use torch.linalg.eigh for batched symmetric eigenvalue decomposition
    # This is much faster than LOBPCG on individual matrices in a loop on GPU
    eigenvalues_full, eigenvectors_full = torch.linalg.eigh(laplacians)
    
    # Extract top k
    all_eigenvalues = eigenvalues_full[:, :k_eigenvalues].to(dtype=dtype)
    all_eigenvectors = eigenvectors_full[:, :, :k_eigenvalues].to(dtype=dtype)

    # 5. Graph Creation (Optional, kept sequential as igraph doesn't support batching well)
    all_graphs = [] if compute_graph_features else None
    if compute_graph_features:
        for b in range(b_size):
            attention = attention_sparse[b]
            nz_mask = attention > 0
            nz_indices = torch.nonzero(nz_mask, as_tuple=False)
            nz_values = attention[nz_mask]
            
            i_upper = nz_indices[:, 0]
            j_upper = nz_indices[:, 1] + n_q
            i_lower = nz_indices[:, 1] + n_q
            j_lower = nz_indices[:, 0]
            
            i_indices = torch.cat([i_upper, i_lower], dim=0)
            j_indices = torch.cat([j_upper, j_lower], dim=0)
            values = torch.cat([nz_values, nz_values], dim=0)

            if values.numel() > 0:
                edges = list(zip(i_indices.cpu().numpy(), j_indices.cpu().numpy()))
                weights = values.cpu().numpy()
                G = ig.Graph(n=n_total, edges=edges, directed=False)
                G.es['weight'] = weights.tolist()
            else:
                G = ig.Graph(n=n_total, directed=False)
                
            # Compute degrees
            degrees_unweighted = torch.bincount(i_indices, minlength=n_total).float()
            degree_centrality_val = degrees_unweighted / (n_total - 1) if n_total > 1 else torch.zeros(n_total, device=device_obj)
            
            # Compute eigenvector centrality
            b_matrix_sparse = torch.sparse_coo_tensor(
                torch.stack([i_indices, j_indices], dim=0), 
                values, 
                (n_total, n_total),
                dtype=computation_dtype, device=device_obj
            ).coalesce()
            
            v = torch.ones(n_total, 1, device=device_obj, dtype=computation_dtype)
            for _ in range(20):
                v = torch.sparse.mm(b_matrix_sparse, v)
                norm = torch.norm(v)
                if norm > 1e-9:
                    v = v / norm
                else:
                    break
            eigenvector_centrality_val = v.squeeze()

            G.degree_centrality_val = degree_centrality_val.cpu().numpy()
            G.eigenvector_centrality_val = eigenvector_centrality_val.float().cpu().numpy()
            all_graphs.append(G)

    if reshape_output:
        b, h = original_shape
        all_eigenvalues = all_eigenvalues.view(b, h, k_eigenvalues)
        all_eigenvectors = all_eigenvectors.view(b, h, n_total, k_eigenvalues)

    return all_eigenvalues, all_eigenvectors, all_graphs