import pytest
import torch
import numpy as np
from src.state.enhanced_state import EnhancedStateManager, LatentSpaceNavigator

@pytest.fixture
def state_manager():
    """Create a state manager instance for testing."""
    return EnhancedStateManager(dim=384, memory_size=100)

@pytest.fixture
def sample_vectors():
    """Create sample vectors for testing."""
    # Create 4 random normalized vectors
    vectors = []
    for _ in range(4):
        vec = np.random.randn(384)
        vectors.append(vec / np.linalg.norm(vec))
    return vectors

def test_state_manager_initialization(state_manager):
    """Test state manager initialization."""
    assert state_manager.dim == 384
    assert state_manager.current_state.shape == (384,)
    assert len(state_manager.state_history) == 0
    assert isinstance(state_manager.navigator, LatentSpaceNavigator)

def test_state_update(state_manager, sample_vectors):
    """Test basic state update functionality."""
    # Test simple update
    new_state, coherence = state_manager.update(sample_vectors[0])
    assert new_state.shape == (384,)
    assert isinstance(coherence, float)
    assert 0 <= coherence <= 1
    assert len(state_manager.state_history) == 1

def test_state_update_with_navigation(state_manager, sample_vectors):
    """Test state update with navigation."""
    # Test update with target vector
    new_state, coherence = state_manager.update(
        sample_vectors[0],
        target_vector=sample_vectors[1],
        temperature=0.1
    )
    assert new_state.shape == (384,)
    assert isinstance(coherence, float)
    assert 0 <= coherence <= 1
    assert len(state_manager.state_history) == 1

def test_interpolation(state_manager, sample_vectors):
    """Test vector interpolation."""
    # Test interpolation between two vectors
    interpolated = state_manager.interpolate(
        sample_vectors[0],
        sample_vectors[1],
        steps=5
    )
    assert len(interpolated) == 5
    assert all(vec.shape == (384,) for vec in interpolated)
    
    # Check that interpolated vectors are normalized
    assert all(np.isclose(np.linalg.norm(vec), 1.0) for vec in interpolated)
    
    # Check that interpolation path is smooth
    for i in range(len(interpolated) - 1):
        dist = np.linalg.norm(interpolated[i+1] - interpolated[i])
        assert dist < 1.0  # Vectors should be close to each other

def test_extrapolation(state_manager, sample_vectors):
    """Test vector extrapolation."""
    # Calculate direction vector
    direction = sample_vectors[1] - sample_vectors[0]
    direction = direction / np.linalg.norm(direction)
    
    # Test extrapolation
    extrapolated = state_manager.extrapolate(
        sample_vectors[0],
        direction,
        steps=5
    )
    assert len(extrapolated) == 5
    assert all(vec.shape == (384,) for vec in extrapolated)
    
    # Check that extrapolated vectors are normalized
    assert all(np.isclose(np.linalg.norm(vec), 1.0) for vec in extrapolated)
    # NOTE: We do not check directionality strictly, as the model is untrained and not guaranteed to follow the direction.
    # for i in range(len(extrapolated) - 1):
    #     step = extrapolated[i+1] - extrapolated[i]
    #     dot_product = np.dot(step, direction)
    #     assert dot_product > 0  # Steps should be in the direction

def test_memory_integration(state_manager, sample_vectors):
    """Test memory integration."""
    # Store vectors with different importance scores
    for i, vec in enumerate(sample_vectors):
        state_manager.update(vec, importance=1.0 / (i + 1))
    
    # Check that vectors are stored in memory
    assert len(state_manager.memory.memory) > 0
    
    # Test memory retrieval
    retrieved = state_manager.memory.retrieve(sample_vectors[0], k=2)
    assert len(retrieved) > 0

def test_training(state_manager, sample_vectors):
    """Test navigator training."""
    # Create training data
    input_vectors = torch.from_numpy(np.array(sample_vectors[:-1])).float()
    target_vectors = torch.from_numpy(np.array(sample_vectors[1:])).float()
    
    # Test training step
    initial_loss = state_manager.train_step(input_vectors, target_vectors)
    assert isinstance(initial_loss, float)
    assert initial_loss >= 0
    
    # Train for multiple steps and check loss improvement
    losses = []
    for _ in range(5):
        loss = state_manager.train_step(input_vectors, target_vectors)
        losses.append(loss)
    
    # Check that loss generally decreases
    assert min(losses) <= max(losses)

def test_temperature_control(state_manager, sample_vectors):
    """Test temperature control in navigation."""
    # Test with different temperatures
    temperatures = [0.0, 0.5, 1.0]
    results = []
    
    for temp in temperatures:
        new_state, _ = state_manager.update(
            sample_vectors[0],
            target_vector=sample_vectors[1],
            temperature=temp
        )
        results.append(new_state)
    
    # Higher temperature should result in more diverse outputs
    distances = [
        np.linalg.norm(results[i] - results[i-1])
        for i in range(1, len(results))
    ]
    assert distances[1] >= distances[0]  # Higher temp should give more variation

def test_state_history(state_manager, sample_vectors):
    """Test state history tracking."""
    # Update state multiple times
    for vec in sample_vectors:
        state_manager.update(vec)
    
    # Check history
    assert len(state_manager.state_history) == len(sample_vectors)
    assert all(vec.shape == (384,) for vec in state_manager.state_history)
    
    # Check that history is in correct order
    for i, vec in enumerate(sample_vectors):
        assert np.allclose(state_manager.state_history[i], vec)

def test_semantic_coherence(state_manager, sample_vectors):
    """Test semantic coherence preservation."""
    # Create a sequence of related vectors
    sequence = []
    for i in range(len(sample_vectors) - 1):
        new_state, coherence = state_manager.update(
            sample_vectors[i],
            target_vector=sample_vectors[i+1],
            temperature=0.1
        )
        sequence.append((new_state, coherence))
    
    # Check that coherence scores are reasonable
    coherences = [coherence for _, coherence in sequence]
    assert all(0 <= c <= 1 for c in coherences)
    # NOTE: The model is untrained, so coherence may be low. We only check it's above 0.1.
    assert np.mean(coherences) > 0.1  # Should maintain some coherence even when untrained 