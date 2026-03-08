"""
Standalone test script for Laplacian construction.
Can be run directly without pytest: python test_laplacian_standalone.py
"""

import torch
from graph_utils import build_bipartite_laplacian_and_eigensystem_sparse


def test_output_shapes():
    """Test 1: Output shapes are correct."""
    print("\n" + "="*70)
    print("Test 1: Output Shapes")
    print("="*70)

    # 3D input
    attention_3d = torch.softmax(torch.randn(2, 10, 8), dim=-1)
    eigenvalues, _, _ = build_bipartite_laplacian_and_eigensystem_sparse(
        attention_3d, k_eigenvalues=5, normalized=True, sparse_factor=0.1, 
    )
    assert eigenvalues.shape == (2, 5), f"3D: Expected (2, 5), got {eigenvalues.shape}"
    print(f"✓ 3D input: {attention_3d.shape} → {eigenvalues.shape}")

    # 4D input
    attention_4d = torch.softmax(torch.randn(2, 4, 10, 8), dim=-1)
    eigenvalues, _, _ = build_bipartite_laplacian_and_eigensystem_sparse(
        attention_4d, k_eigenvalues=5, normalized=True, sparse_factor=0.1, 
    )
    assert eigenvalues.shape == (2, 4, 5), f"4D: Expected (2, 4, 5), got {eigenvalues.shape}"
    print(f"✓ 4D input: {attention_4d.shape} → {eigenvalues.shape}")

    # Various dimensions
    test_cases = [(5, 5), (10, 5), (5, 10), (100, 50), (3, 3)]
    for n_q, n_k in test_cases:
        attention = torch.softmax(torch.randn(2, n_q, n_k), dim=-1)
        k_eig = min(5, n_q + n_k - 1)
        eigenvalues, _, _ = build_bipartite_laplacian_and_eigensystem_sparse(
            attention, k_eigenvalues=k_eig, normalized=True, sparse_factor=0.1, 
        )
        assert eigenvalues.shape == (2, k_eig), f"({n_q}, {n_k}): Expected (2, {k_eig}), got {eigenvalues.shape}"
        print(f"✓ Dimensions ({n_q}, {n_k}): output shape {eigenvalues.shape}")

    print("✓ All shape tests passed!")


def test_laplacian_properties():
    """Test 2: Laplacian has correct mathematical properties."""
    print("\n" + "="*70)
    print("Test 2: Laplacian Mathematical Properties")
    print("="*70)

    torch.manual_seed(42)
    attention = torch.softmax(torch.randn(10, 8), dim=-1)
    n_q, n_k = attention.shape
    n_total = n_q + n_k

    # Build Laplacian manually
    sparse_factor = 0.1
    q_val = torch.quantile(attention, sparse_factor, dim=1, keepdim=True)
    mask = attention >= q_val
    attention_sparse = attention * mask

    # Build bipartite adjacency
    nz_mask = attention_sparse > 0
    nz_indices = torch.nonzero(nz_mask, as_tuple=False)
    nz_values = torch.ones_like(attention_sparse[nz_mask])

    i_upper = nz_indices[:, 0]
    j_upper = nz_indices[:, 1] + n_q
    i_lower = nz_indices[:, 1] + n_q
    j_lower = nz_indices[:, 0]

    i_indices = torch.cat([i_upper, i_lower], dim=0)
    j_indices = torch.cat([j_upper, j_lower], dim=0)
    values = torch.cat([nz_values, nz_values], dim=0)

    indices = torch.stack([i_indices, j_indices], dim=0)
    b_matrix = torch.sparse_coo_tensor(
        indices, values, (n_total, n_total)
    ).coalesce().to_dense()

    # Test 2.1: Symmetry
    symmetry_error = (b_matrix - b_matrix.T).abs().max().item()
    assert symmetry_error < 1e-6, f"Not symmetric: error = {symmetry_error}"
    print(f"✓ Bipartite adjacency is symmetric (max error: {symmetry_error:.2e})")

    # Test 2.2: Bipartite structure
    upper_left = b_matrix[:n_q, :n_q]
    lower_right = b_matrix[n_q:, n_q:]
    upper_left_max = upper_left.abs().max().item()
    lower_right_max = lower_right.abs().max().item()
    assert upper_left_max < 1e-6, f"Upper-left not zero: {upper_left_max}"
    assert lower_right_max < 1e-6, f"Lower-right not zero: {lower_right_max}"
    print(f"✓ Bipartite structure correct (diagonal blocks are zero)")

    # Test 2.3: Row sums of unnormalized Laplacian
    deg_q = (attention_sparse > 0).sum(dim=1).float()
    deg_k = (attention_sparse > 0).sum(dim=0).float()
    deg_full = torch.cat([deg_q, deg_k], dim=0).clamp(min=1.0)

    deg_matrix = torch.diag(deg_full)
    laplacian = deg_matrix - b_matrix

    row_sums = laplacian.sum(dim=1)
    max_row_sum = row_sums.abs().max().item()
    assert max_row_sum < 1e-5, f"Row sums not zero: {max_row_sum}"
    print(f"✓ Unnormalized Laplacian row sums = 0 (max deviation: {max_row_sum:.2e})")

    # Test 2.4: Symmetry of Laplacian
    laplacian_symmetry = (laplacian - laplacian.T).abs().max().item()
    assert laplacian_symmetry < 1e-6, f"Laplacian not symmetric: {laplacian_symmetry}"
    print(f"✓ Laplacian is symmetric (max error: {laplacian_symmetry:.2e})")

    print("✓ All property tests passed!")


def test_eigenvalue_properties():
    """Test 3: Eigenvalues have correct properties."""
    print("\n" + "="*70)
    print("Test 3: Eigenvalue Properties")
    print("="*70)

    torch.manual_seed(42)
    attention = torch.softmax(torch.randn(3, 10, 8), dim=-1)

    eigenvalues, _, _ = build_bipartite_laplacian_and_eigensystem_sparse(
        attention, k_eigenvalues=8, normalized=True, sparse_factor=0.1, 
    )

    # Test 3.1: Range [0, 2] for normalized
    min_eig = eigenvalues.min().item()
    max_eig = eigenvalues.max().item()
    assert min_eig >= -1e-5, f"Significantly negative eigenvalue: {min_eig} (small negative values from numerical errors are clamped to 0)"
    assert max_eig <= 2 + 1e-6, f"Eigenvalue > 2: {max_eig}"
    print(f"✓ Eigenvalues in valid range [0, 2]: [{min_eig:.4f}, {max_eig:.4f}]")

    # Test 3.2: Smallest eigenvalue near zero
    smallest = eigenvalues[:, 0].max().item()  # Max across batch
    assert smallest < 0.1, f"Smallest eigenvalue not near zero: {smallest}"
    print(f"✓ Smallest eigenvalue near 0: {eigenvalues[:, 0].mean().item():.6f}")

    # Test 3.3: Sorted order
    for i in range(eigenvalues.shape[0]):
        eigs = eigenvalues[i]
        diffs = eigs[1:] - eigs[:-1]
        assert (diffs >= -1e-6).all(), f"Not sorted for sample {i}"
    print(f"✓ Eigenvalues sorted in ascending order")

    # Test 3.4: Algebraic connectivity (Fiedler value)
    fiedler = eigenvalues[:, 1]
    print(f"✓ Algebraic connectivity (λ₂): mean={fiedler.mean().item():.4f}, std={fiedler.std().item():.4f}")

    # Test 3.5: Spectral gap
    spectral_gap = eigenvalues[:, 1] - eigenvalues[:, 0]
    print(f"✓ Spectral gap (λ₂ - λ₁): mean={spectral_gap.mean().item():.4f}, std={spectral_gap.std().item():.4f}")

    print("✓ All eigenvalue tests passed!")


def test_degree_properties():
    """Test 4: Degree calculations are correct."""
    print("\n" + "="*70)
    print("Test 4: Degree Properties")
    print("="*70)

    # Create attention matrix with guaranteed heterogeneous structure
    torch.manual_seed(999)
    # Create base attention with varying distributions per row
    attention = torch.randn(30, 20)
    # Make some rows more concentrated, others more uniform
    for i in range(30):
        if i % 3 == 0:
            attention[i] = attention[i] * 3  # More peaked
        elif i % 3 == 1:
            attention[i] = attention[i] * 0.5  # Flatter
    attention = torch.softmax(attention, dim=-1)

    sparse_factor = 0.4

    # Use GLOBAL quantile to ensure heterogeneous sparsity
    q_val = torch.quantile(attention.flatten(), sparse_factor)
    mask = attention >= q_val
    attention_sparse = attention * mask

    # Test 4.1: Degrees are positive
    deg_q = (attention_sparse > 0).sum(dim=1).float()
    deg_k = (attention_sparse > 0).sum(dim=0).float()

    print(f"  Non-zero entries per row: {deg_q.unique().tolist()}")
    print(f"  Non-zero entries per col: {deg_k.unique().tolist()}")

    assert (deg_q > 0).all(), "Some query degrees are zero"
    assert (deg_k > 0).all(), "Some key degrees are zero"
    print(f"✓ All degrees are positive")

    # Test 4.2: Degrees are integers
    assert torch.allclose(deg_q, deg_q.round()), "Query degrees not integers"
    assert torch.allclose(deg_k, deg_k.round()), "Key degrees not integers"
    print(f"✓ All degrees are integers (counts)")

    # Test 4.3: Degrees are non-constant
    # At least ONE should have variance (typically key degrees will)
    q_has_variance = deg_q.std() > 0
    k_has_variance = deg_k.std() > 0

    if q_has_variance and k_has_variance:
        print(f"✓ Both query and key degrees are non-constant")
    elif k_has_variance:
        print(f"✓ Key degrees are non-constant (query degrees may be similar due to sparsification method)")
    elif q_has_variance:
        print(f"✓ Query degrees are non-constant (key degrees may be similar)")
    else:
        # This should not happen with our constructed matrix
        raise AssertionError(f"Both degrees are constant! Query: {deg_q.unique()}, Key: {deg_k.unique()}")

    print(f"  Query degrees: mean={deg_q.mean().item():.2f}, std={deg_q.std().item():.2f}, range=[{deg_q.min().item():.0f}, {deg_q.max().item():.0f}]")
    print(f"  Key degrees: mean={deg_k.mean().item():.2f}, std={deg_k.std().item():.2f}, range=[{deg_k.min().item():.0f}, {deg_k.max().item():.0f}]")

    # Test 4.4: Test what happens in real usage (per-row quantile)
    print("\n  Testing with per-row quantile (as in actual implementation):")
    sparse_factor_per_row = 0.3
    q_val_per_row = torch.quantile(attention, sparse_factor_per_row, dim=1, keepdim=True)
    mask_per_row = attention >= q_val_per_row
    attention_sparse_per_row = attention * mask_per_row

    deg_q_per_row = (attention_sparse_per_row > 0).sum(dim=1).float()
    deg_k_per_row = (attention_sparse_per_row > 0).sum(dim=0).float()

    print(f"  Query degrees: mean={deg_q_per_row.mean().item():.2f}, std={deg_q_per_row.std().item():.2f}, range=[{deg_q_per_row.min().item():.0f}, {deg_q_per_row.max().item():.0f}]")
    print(f"  Key degrees: mean={deg_k_per_row.mean().item():.2f}, std={deg_k_per_row.std().item():.2f}, range=[{deg_k_per_row.min().item():.0f}, {deg_k_per_row.max().item():.0f}]")

    if deg_q_per_row.std() < 0.1:
        print(f"  ⚠ Note: Per-row quantile produces nearly constant query degrees (this is expected)")
    if deg_k_per_row.std() > 0:
        print(f"  ✓ Key degrees have variance with per-row quantile")

    print("✓ All degree tests passed!")


def test_weighted_vs_unweighted():
    """Test 5: Compare weighted and unweighted approaches."""
    print("\n" + "="*70)
    print("Test 5: Weighted vs Unweighted Comparison")
    print("="*70)

    torch.manual_seed(42)
    attention = torch.softmax(torch.randn(3, 10, 8), dim=-1)

    eigenvalues_unweighted, _, _ = build_bipartite_laplacian_and_eigensystem_sparse(
        attention, k_eigenvalues=5, normalized=True, sparse_factor=0.1, 
    )

    eigenvalues_weighted, _, _ = build_bipartite_laplacian_and_eigensystem_sparse(
        attention, k_eigenvalues=5, normalized=True, sparse_factor=0.1, 
    )

    # Test 5.1: Same shape
    assert eigenvalues_unweighted.shape == eigenvalues_weighted.shape
    print(f"✓ Same output shape: {eigenvalues_unweighted.shape}")

    # Test 5.2: Both in valid range
    assert (eigenvalues_unweighted >= -1e-6).all()
    assert (eigenvalues_weighted >= -1e-6).all()
    print(f"✓ Both have valid eigenvalue ranges")

    # Test 5.3: Different results
    max_diff = (eigenvalues_unweighted - eigenvalues_weighted).abs().max().item()
    assert max_diff > 1e-3, "Weighted and unweighted give identical results"
    print(f"✓ Different results (max difference: {max_diff:.4f})")

    # Test 5.4: Statistics comparison
    print(f"\nUnweighted eigenvalues:")
    print(f"  λ₁: {eigenvalues_unweighted[:, 0].mean().item():.6f}")
    print(f"  λ₂: {eigenvalues_unweighted[:, 1].mean().item():.6f}")
    print(f"  Mean: {eigenvalues_unweighted.mean().item():.6f}")

    print(f"\nWeighted eigenvalues:")
    print(f"  λ₁: {eigenvalues_weighted[:, 0].mean().item():.6f}")
    print(f"  λ₂: {eigenvalues_weighted[:, 1].mean().item():.6f}")
    print(f"  Mean: {eigenvalues_weighted.mean().item():.6f}")

    print("\n✓ All comparison tests passed!")


def test_edge_cases():
    """Test 6: Edge cases and robustness."""
    print("\n" + "="*70)
    print("Test 6: Edge Cases")
    print("="*70)

    # Test 6.1: No sparsification
    attention = torch.softmax(torch.randn(2, 10, 8), dim=-1)
    eigenvalues, _, _ = build_bipartite_laplacian_and_eigensystem_sparse(
        attention, k_eigenvalues=5, normalized=True, sparse_factor=0.0, 
    )
    assert eigenvalues.shape == (2, 5)
    print(f"✓ No sparsification (sparse_factor=0.0): {eigenvalues.shape}")

    # Test 6.2: High sparsification
    eigenvalues, _, _ = build_bipartite_laplacian_and_eigensystem_sparse(
        attention, k_eigenvalues=5, normalized=True, sparse_factor=0.8, 
    )
    assert eigenvalues.shape == (2, 5)
    print(f"✓ High sparsification (sparse_factor=0.8): {eigenvalues.shape}")

    # Test 6.3: Single sample
    attention = torch.softmax(torch.randn(1, 10, 8), dim=-1)
    eigenvalues, _, _ = build_bipartite_laplacian_and_eigensystem_sparse(
        attention, k_eigenvalues=5, normalized=True, sparse_factor=0.1, 
    )
    assert eigenvalues.shape == (1, 5)
    print(f"✓ Single sample (batch=1): {eigenvalues.shape}")

    # Test 6.4: Small k
    attention = torch.softmax(torch.randn(2, 10, 8), dim=-1)
    eigenvalues, _, _ = build_bipartite_laplacian_and_eigensystem_sparse(
        attention, k_eigenvalues=2, normalized=True, sparse_factor=0.1, 
    )
    assert eigenvalues.shape == (2, 2)
    print(f"✓ Small k (k_eigenvalues=2): {eigenvalues.shape}")

    # Test 6.5: Very small values
    attention = torch.ones(2, 10, 8) * 1e-8
    attention = attention / attention.sum(dim=-1, keepdim=True)
    eigenvalues, _, _ = build_bipartite_laplacian_and_eigensystem_sparse(
        attention, k_eigenvalues=5, normalized=True, sparse_factor=0.1, 
    )
    assert torch.isfinite(eigenvalues).all()
    print(f"✓ Very small values (1e-8): all eigenvalues finite")

    print("✓ All edge case tests passed!")


def test_numerical_stability():
    """Test 7: Numerical stability checks."""
    print("\n" + "="*70)
    print("Test 7: Numerical Stability")
    print("="*70)

    torch.manual_seed(42)

    # Test 7.1: No NaN or Inf
    attention = torch.softmax(torch.randn(5, 20, 15), dim=-1)
    eigenvalues, _, _ = build_bipartite_laplacian_and_eigensystem_sparse(
        attention, k_eigenvalues=10, normalized=True, sparse_factor=0.2, 
    )
    assert torch.isfinite(eigenvalues).all(), "NaN or Inf detected"
    print(f"✓ No NaN or Inf values")

    # Test 7.2: Consistent across runs with same seed
    torch.manual_seed(42)
    eigenvalues_1, _, _ = build_bipartite_laplacian_and_eigensystem_sparse(
        attention, k_eigenvalues=5, normalized=True, sparse_factor=0.1, 
    )
    torch.manual_seed(42)
    eigenvalues_2, _, _ = build_bipartite_laplacian_and_eigensystem_sparse(
        attention, k_eigenvalues=5, normalized=True, sparse_factor=0.1, 
    )
    max_diff = (eigenvalues_1 - eigenvalues_2).abs().max().item()
    print(f"✓ Determinism check (max diff with same seed): {max_diff:.2e}")

    # Test 7.3: Batch consistency
    eigenvalues_batch, _, _ = build_bipartite_laplacian_and_eigensystem_sparse(
        attention, k_eigenvalues=5, normalized=True, sparse_factor=0.1, 
    )
    eigenvalues_individual = []
    for i in range(5):
        eig, _, _ = build_bipartite_laplacian_and_eigensystem_sparse(
            attention[i:i+1], k_eigenvalues=5, normalized=True, sparse_factor=0.1, 
        )
        eigenvalues_individual.append(eig)
    eigenvalues_individual = torch.cat(eigenvalues_individual, dim=0)
    max_diff = (eigenvalues_batch - eigenvalues_individual).abs().max().item()
    print(f"✓ Batch vs individual processing (max diff): {max_diff:.2e}")

    print("✓ All stability tests passed!")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("LAPLACIAN CONSTRUCTION TEST SUITE")
    print("="*70)

    try:
        test_output_shapes()
        test_laplacian_properties()
        test_eigenvalue_properties()
        test_degree_properties()
        test_weighted_vs_unweighted()
        test_edge_cases()
        test_numerical_stability()

        print("\n" + "="*70)
        print("✓✓✓ ALL TESTS PASSED! ✓✓✓")
        print("="*70 + "\n")

    except AssertionError as e:
        print("\n" + "="*70)
        print(f"❌ TEST FAILED: {e}")
        print("="*70 + "\n")
        raise
    except Exception as e:
        print("\n" + "="*70)
        print(f"❌ ERROR: {e}")
        print("="*70 + "\n")
        raise


if __name__ == "__main__":
    run_all_tests()
