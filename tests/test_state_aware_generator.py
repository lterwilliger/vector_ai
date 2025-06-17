import pytest
import numpy as np
import torch
import os
from src.generation.state_aware_generator import StateAwareGenerator, StateAwareGenerationSystem

def test_state_aware_generator():
    dim = 64
    batch_size = 2
    seq_len = 3
    generator = StateAwareGenerator(dim)
    
    # Create test tensors
    state = torch.randn(batch_size, dim)
    memory_vectors = torch.randn(batch_size, seq_len, dim)  # (batch, seq_len, dim)
    
    # Test forward pass
    output = generator(state, memory_vectors)
    
    assert output.shape == (batch_size, dim)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()
    
    # Test normalization
    norms = torch.norm(output, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

def test_generation_system():
    dim = 64
    system = StateAwareGenerationSystem(dim)
    
    # Test generation
    vectors, states = system.generate(temperature=0.5, num_samples=3)
    
    assert vectors.shape == (3, dim)
    assert states.shape == (3, dim)
    assert not np.isnan(vectors).any()
    assert not np.isnan(states).any()
    
    # Test that generated vectors are normalized
    for vector in vectors:
        assert np.isclose(np.linalg.norm(vector), 1.0)
    
    # Test that states are normalized
    for state in states:
        assert np.isclose(np.linalg.norm(state), 1.0)

def test_training():
    dim = 64
    system = StateAwareGenerationSystem(dim)
    
    # Create target vectors
    target_vectors = np.random.randn(2, dim)
    target_vectors = target_vectors / np.linalg.norm(target_vectors, axis=1, keepdims=True)
    
    # Test training step
    loss = system.train_step(target_vectors)
    
    assert isinstance(loss, float)
    assert not np.isnan(loss)
    assert not np.isinf(loss)
    
    # Test that state was updated
    current_state = system.state_manager.get_state()
    assert current_state.shape == (dim,)
    assert not np.isnan(current_state).any()

def test_temperature_effect():
    dim = 64
    system = StateAwareGenerationSystem(dim)
    
    # Test generation with different temperatures
    vectors1, _ = system.generate(temperature=0.0, num_samples=2)
    vectors2, _ = system.generate(temperature=1.0, num_samples=2)
    
    # Higher temperature should result in more variation
    assert not np.array_equal(vectors1, vectors2)

def test_state_persistence():
    dim = 64
    system = StateAwareGenerationSystem(dim)
    
    # Generate some vectors to create state
    system.generate(num_samples=3)
    
    # Save state
    save_path = "test_state.pt"
    system.save_state(save_path)
    
    # Create new system and load state
    new_system = StateAwareGenerationSystem(dim)
    new_system.load_state(save_path)
    
    # Compare states
    original_state = system.state_manager.get_state()
    loaded_state = new_system.state_manager.get_state()
    
    assert np.allclose(original_state, loaded_state)
    
    # Clean up
    os.remove(save_path)

def test_memory_integration():
    dim = 64
    system = StateAwareGenerationSystem(dim)
    
    # Generate initial vectors
    vectors1, _ = system.generate(num_samples=2)
    
    # Update memory importance
    for vector in vectors1:
        system.state_manager.update_memory_importance(vector, importance=2.0)
    
    # Generate new vectors
    vectors2, _ = system.generate(num_samples=2)
    
    # Test that memory affected generation
    assert not np.array_equal(vectors1, vectors2)
    
    # Test memory retrieval
    memory_vectors = system.state_manager.get_memory_vectors(k=2)
    assert len(memory_vectors) == 2
    assert all(vec.shape == (dim,) for vec in memory_vectors)

def test_device_handling():
    dim = 64
    
    # Test CPU
    system_cpu = StateAwareGenerationSystem(dim, device='cpu')
    vectors_cpu, _ = system_cpu.generate()
    assert vectors_cpu.shape == (1, dim)
    
    # Test CUDA if available
    if torch.cuda.is_available():
        system_cuda = StateAwareGenerationSystem(dim, device='cuda')
        vectors_cuda, _ = system_cuda.generate()
        assert vectors_cuda.shape == (1, dim) 