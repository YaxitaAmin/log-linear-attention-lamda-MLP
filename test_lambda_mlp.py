import torch
from hattention.lambda_mlp import LambdaMLPSoftplus, LambdaMLPSoftmax

def test_softplus():
    print("Testing LambdaMLPSoftplus...")
    batch, seqlen, nheads, hidden_dim, num_levels = 2, 16, 4, 64, 12

    mlp = LambdaMLPSoftplus(
        num_levels=num_levels,
        hidden_dim=hidden_dim)

    x = torch.randn(batch, seqlen, nheads, num_levels)
    out = mlp(x)

    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {out.shape}")
    assert out.shape == (batch, seqlen, nheads, num_levels), "Shape mismatch!"
    assert (out > 0).all(), "Softplus output should be positive!"
    print(f"  Output mean: {out.mean().item():.4f} (should be close to 1.0 at init)")
    print("  PASSED! ✅")


def test_softmax():
    print("Testing LambdaMLPSoftmax...")
    batch, seqlen, nheads, hidden_dim, num_levels = 2, 16, 4, 64, 12

    mlp = LambdaMLPSoftmax(
        num_levels=num_levels,
        hidden_dim=hidden_dim)

    x = torch.randn(batch, seqlen, nheads, num_levels)
    out = mlp(x)

    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {out.shape}")
    assert out.shape == (batch, seqlen, nheads, num_levels), "Shape mismatch!"
    assert (out > 0).all(), "Softmax output should be positive!"
    # Softmax sums to 1 across levels
    sums = out.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), "Softmax should sum to 1!"
    print(f"  Output sum across levels: {sums.mean().item():.4f} (should be 1.0)")
    print("  PASSED! ✅")


def test_identity_init():
    print("Testing identity initialization...")
    batch, seqlen, nheads, hidden_dim, num_levels = 1, 8, 2, 64, 12

    mlp = LambdaMLPSoftplus(
        num_levels=num_levels,
        hidden_dim=hidden_dim)

    x = torch.zeros(batch, seqlen, nheads, num_levels)
    out = mlp(x)

    print(f"  Output at zero input: {out.mean().item():.4f} (should be ≈ 1.0)")
    assert abs(out.mean().item() - 1.0) < 0.1, "Identity init failed!"
    print("  PASSED! ✅")


if __name__ == "__main__":
    test_softplus()
    print()
    test_softmax()
    print()
    test_identity_init()
    print()
    print("All tests passed! 🎉")