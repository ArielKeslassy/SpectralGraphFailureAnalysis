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

    all_eigenvalues = torch.zeros(b_size, k_eigenvalues, device=device_obj, dtype=dtype)
    all_eigenvectors = torch.zeros(b_size, n_total, k_eigenvalues, device=device_obj, dtype=dtype)
    all_graphs = [] if compute_graph_features else None

    # OPTIMIZATION: Batch sparsification
    if sparse_factor > 0:
        q_val = torch.quantile(attention_matrix.float(), sparse_factor, dim=2, keepdim=True)
        mask = attention_matrix >= q_val
        attention_sparse = attention_matrix * mask
    else:
        attention_sparse = attention_matrix

    for b in range(b_size):
        attention = attention_sparse[b].to(dtype=computation_dtype)  # [n_q, n_k]

        # 2. Build sparse bipartite adjacency matrix components
        nz_mask = attention > 0
        nz_indices = torch.nonzero(nz_mask, as_tuple=False)  # [nnz, 2]
        nz_values = attention[nz_mask]  # [nnz]

        # 3. Calculate Normalized Out-Degree
        received_attention = torch.sum(attention, dim=0)  # [n_k]
        indices_k = torch.arange(n_k, device=device_obj)
        denominator = torch.clamp(n_q - indices_k, min=1.0)
        deg_k = received_attention / denominator
        deg_q = torch.ones(n_q, device=device_obj, dtype=computation_dtype)
        deg = torch.cat([deg_q, deg_k], dim=0)
        deg = torch.clamp(deg, min=1e-12)

        # 4. Build sparse bipartite adjacency matrix
        i_upper = nz_indices[:, 0]
        j_upper = nz_indices[:, 1] + n_q
        i_lower = nz_indices[:, 1] + n_q
        j_lower = nz_indices[:, 0]

        i_indices = torch.cat([i_upper, i_lower], dim=0)
        j_indices = torch.cat([j_upper, j_lower], dim=0)
        values = torch.cat([nz_values, nz_values], dim=0)

        indices = torch.stack([i_indices, j_indices], dim=0)
        b_matrix_sparse = torch.sparse_coo_tensor(
            indices, values, (n_total, n_total),
            dtype=computation_dtype, device=device_obj
        ).coalesce()

        # OPTIMIZATION: Compute all centrality metrics on GPU in batch
        degrees_unweighted = torch.bincount(indices[0], minlength=n_total).float()
        degree_centrality_val = degrees_unweighted / (n_total - 1) if n_total > 1 else torch.zeros(n_total,
                                                                                                   device=device_obj)

        # Eigenvector Centrality via power iteration (on GPU/MPS)
        v = torch.ones(n_total, 1, device=device_obj, dtype=computation_dtype)
        for _ in range(20):
            v = torch.sparse.mm(b_matrix_sparse, v)
            norm = torch.norm(v)
            if norm > 1e-9:
                v = v / norm
            else:
                break
        eigenvector_centrality_val = v.squeeze()

        # OPTIMIZATION: Only create graph object if needed
        if compute_graph_features:
            if values.numel() > 0:
                # Single CPU transfer for graph creation
                edges = list(zip(i_indices.cpu().numpy(), j_indices.cpu().numpy()))
                weights = values.cpu().numpy()
                G = ig.Graph(n=n_total, edges=edges, directed=False)
                G.es['weight'] = weights.tolist()
            else:
                G = ig.Graph(n=n_total, directed=False)

            # Store pre-computed centrality as numpy arrays (already computed on GPU)
            G.degree_centrality_val = degree_centrality_val.cpu().numpy()
            G.eigenvector_centrality_val = eigenvector_centrality_val.float().cpu().numpy()
            all_graphs.append(G)

        # 5. Construct Laplacian
        if normalized:
            deg_inv_sqrt = torch.pow(deg, -0.5)
            diag_indices = torch.arange(n_total, device=device_obj).unsqueeze(0).repeat(2, 1)
            d_inv_sqrt_sparse = torch.sparse_coo_tensor(
                diag_indices, deg_inv_sqrt, (n_total, n_total),
                dtype=computation_dtype, device=device_obj
            )
            temp = torch.sparse.mm(d_inv_sqrt_sparse, b_matrix_sparse)
            normalized_b = torch.sparse.mm(temp, d_inv_sqrt_sparse)
            laplacian_sparse = normalized_b
            transform_eigenvalues = True
        else:
            diag_indices = torch.arange(n_total, device=device_obj).unsqueeze(0).repeat(2, 1)
            deg_matrix_sparse = torch.sparse_coo_tensor(
                diag_indices, deg, (n_total, n_total),
                dtype=computation_dtype, device=device_obj
            )
            laplacian_sparse = (deg_matrix_sparse - b_matrix_sparse).coalesce()
            transform_eigenvalues = False

        # 6. Eigenvalue computation
        try:
            laplacian_dense = laplacian_sparse.to_dense()
            laplacian_dense = (laplacian_dense + laplacian_dense.T) / 2
            X_init = torch.randn(n_total, k_eigenvalues, dtype=computation_dtype, device=device_obj)

            eigenvalues_raw, eigenvectors = torch.lobpcg(
                -laplacian_dense, k=k_eigenvalues, X=X_init, niter=50, largest=True
            )
            eigenvalues = -eigenvalues_raw

            if transform_eigenvalues:
                eigenvalues = 1 - eigenvalues

            eigenvalues, perm = torch.sort(eigenvalues)
            eigenvectors = eigenvectors[:, perm]

        except Exception:
            # Manual fallback to CPU if MPS fails (e.g. for linalg.eig)
            use_cpu_fallback = (device_obj.type == 'mps')
            
            if use_cpu_fallback:
                laplacian_dense = laplacian_sparse.to_dense().cpu()
            else:
                laplacian_dense = laplacian_sparse.to_dense()

            laplacian_dense = (laplacian_dense + laplacian_dense.T) / 2
            
            current_device = laplacian_dense.device

            if transform_eigenvalues:
                eigenvalues_full, eigenvectors_full = torch.linalg.eig(
                    torch.eye(n_total, device=current_device) - laplacian_dense
                )
                # Take real part if complex, though it should be real for symmetric matrices
                if eigenvalues_full.is_complex():
                    eigenvalues_full = eigenvalues_full.real
                    eigenvectors_full = eigenvectors_full.real
            else:
                eigenvalues_full, eigenvectors_full = torch.linalg.eigh(laplacian_dense)
            
            # Sort eigenvalues to ensure we get the smallest ones
            # linalg.eigh returns them in ascending order, but linalg.eig might not
            if transform_eigenvalues:
                 eigenvalues_full, indices = torch.sort(eigenvalues_full)
                 eigenvectors_full = eigenvectors_full[:, indices]

            eigenvalues = eigenvalues_full[:k_eigenvalues]
            eigenvectors = eigenvectors_full[:, :k_eigenvalues]
            
            if use_cpu_fallback:
                eigenvalues = eigenvalues.to(device_obj)
                eigenvectors = eigenvectors.to(device_obj)

        # OPTIMIZATION: Direct assignment instead of list append + stack
        all_eigenvalues[b] = eigenvalues
        all_eigenvectors[b] = eigenvectors

    if reshape_output:
        b, h = original_shape
        all_eigenvalues = all_eigenvalues.view(b, h, k_eigenvalues)
        all_eigenvectors = all_eigenvectors.view(b, h, n_total, k_eigenvalues)

    return all_eigenvalues, all_eigenvectors, all_graphs
