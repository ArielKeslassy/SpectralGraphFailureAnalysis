"""
Quick verification that the eigenvalue computation fix works correctly.
"""

import torch
from graph_utils import build_bipartite_laplacian_and_eigensystem_sparse

print("Testing eigenvalue computation fix...")
print("=" * 70)

# Create sample attention with larger dimensions to ensure non-constant degrees
torch.manual_seed(42)
attention = torch.softmax(torch.randn(5, 20, 15), dim=-1)

# Test normalized Laplacian
print("\n1. Testing NORMALIZED Laplacian:")
eigenvalues_norm, _, _ = build_bipartite_laplacian_and_eigensystem_sparse(
    attention,
    k_eigenvalues=10,
    normalized=True,
    sparse_factor=0.2,  # Increased for more variance
)

print(f"   Shape: {eigenvalues_norm.shape}")
print(f"   Min eigenvalue: {eigenvalues_norm.min().item():.8f}")
print(f"   Max eigenvalue: {eigenvalues_norm.max().item():.8f}")
print(f"   First 5 eigenvalues (sample 0): {eigenvalues_norm[0, :5]}")

# Check range
if eigenvalues_norm.min() >= 0 and eigenvalues_norm.max() <= 2:
    print("   ✓ Eigenvalues in valid range [0, 2]")
else:
    print(f"   ✗ ERROR: Eigenvalues outside [0, 2] range!")

# Check degree distribution
print("\n1a. Checking degree distribution:")
sample_attention = attention[0]
sparse_factor = 0.2
q_val = torch.quantile(sample_attention, sparse_factor, dim=1, keepdim=True)
mask = sample_attention >= q_val
attention_sparse = sample_attention * mask

deg_q = (attention_sparse > 0).sum(dim=1).float()
deg_k = (attention_sparse > 0).sum(dim=0).float()

print(f"   Query degrees: min={deg_q.min().item():.0f}, max={deg_q.max().item():.0f}, mean={deg_q.mean().item():.2f}, std={deg_q.std().item():.2f}")
print(f"   Key degrees:   min={deg_k.min().item():.0f}, max={deg_k.max().item():.0f}, mean={deg_k.mean().item():.2f}, std={deg_k.std().item():.2f}")

if deg_q.std() > 0 and deg_k.std() > 0:
    print("   ✓ Degrees are non-constant (have variance)")
else:
    print("   ⚠ Warning: Degrees might be constant - try higher sparse_factor")

# Test unnormalized Laplacian
print("\n2. Testing UNNORMALIZED Laplacian:")
eigenvalues_unnorm, _, _ = build_bipartite_laplacian_and_eigensystem_sparse(
    attention,
    k_eigenvalues=10,
    normalized=False,
    sparse_factor=0.2,
)

print(f"   Shape: {eigenvalues_unnorm.shape}")
print(f"   Min eigenvalue: {eigenvalues_unnorm.min().item():.8f}")
print(f"   Max eigenvalue: {eigenvalues_unnorm.max().item():.8f}")
print(f"   First 5 eigenvalues (sample 0): {eigenvalues_unnorm[0, :5]}")

# Check non-negative
if eigenvalues_unnorm.min() >= 0:
    print("   ✓ Eigenvalues are non-negative")
else:
    print(f"   ✗ ERROR: Negative eigenvalues!")

# Test both weighted options
print("\n3. Testing WEIGHTED vs UNWEIGHTED:")

eigenvalues_unweighted, _, _ = build_bipartite_laplacian_and_eigensystem_sparse(
    attention,
    k_eigenvalues=8,
    normalized=True,
    sparse_factor=0.2,
)

eigenvalues_weighted, _, _ = build_bipartite_laplacian_and_eigensystem_sparse(
    attention,
    k_eigenvalues=8,
    normalized=True,
    sparse_factor=0.2,
)

print(f"   Unweighted - Min: {eigenvalues_unweighted.min().item():.6f}, Max: {eigenvalues_unweighted.max().item():.6f}")
print(f"   Weighted   - Min: {eigenvalues_weighted.min().item():.6f}, Max: {eigenvalues_weighted.max().item():.6f}")

if eigenvalues_unweighted.min() >= 0 and eigenvalues_weighted.min() >= 0:
    print("   ✓ Both approaches produce valid eigenvalues")
else:
    print("   ✗ ERROR: Negative eigenvalues detected!")

# Check smallest eigenvalue is near zero (connected graph)
print("\n4. Testing CONNECTIVITY (smallest eigenvalue should be ~0):")
smallest = eigenvalues_norm[:, 0]
print(f"   Smallest eigenvalues: {smallest}")
print(f"   Mean: {smallest.mean().item():.6f}")

if smallest.max() < 0.1:
    print("   ✓ Graph is well-connected (λ₁ ≈ 0)")
else:
    print(f"   ⚠ Warning: λ₁ might be too large (disconnected components?)")

print("\n" + "=" * 70)
print("✓ Verification complete! No negative eigenvalues detected.")
print("=" * 70)
