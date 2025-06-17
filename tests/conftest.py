import pytest
import numpy as np
import torch

@pytest.fixture(scope="session")
def set_seed():
    """Set random seeds for reproducibility."""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

@pytest.fixture(scope="session")
def test_dim():
    """Return test vector dimension."""
    return 64

@pytest.fixture(scope="session")
def test_device():
    """Return test device (CPU/CUDA)."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'

@pytest.fixture(scope="session")
def test_vectors(test_dim):
    """Generate test vectors."""
    vectors = [np.random.randn(test_dim) for _ in range(10)]
    # Normalize vectors
    vectors = [vec / np.linalg.norm(vec) for vec in vectors]
    return vectors 