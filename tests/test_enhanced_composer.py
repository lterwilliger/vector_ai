import pytest
import numpy as np
import torch
from src.composition.enhanced_composer import EnhancedVectorComposer, MultiHeadAttention, StateTransitionNetwork

def test_multi_head_attention():
    dim = 64
    num_heads = 8
    batch_size = 2
    seq_len = 3
    
    attention = MultiHeadAttention(dim, num_heads)
    
    # Create test tensors
    vectors = torch.randn(batch_size, seq_len, dim)
    state = torch.randn(batch_size, dim)
    
    # Test forward pass
    output = attention(vectors, state)
    
    assert output.shape == (batch_size, dim)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

def test_state_transition_network():
    dim = 64
    batch_size = 2
    
    network = StateTransitionNetwork(dim)
    
    # Create test tensors
    vector = torch.randn(batch_size, dim)
    state = torch.randn(batch_size, dim)
    
    # Test forward pass
    new_state = network(vector, state)
    
    assert new_state.shape == (batch_size, dim)
    assert not torch.isnan(new_state).any()
    assert not torch.isinf(new_state).any()
    
    # Test normalization
    norms = torch.norm(new_state, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

def test_enhanced_vector_composer():
    dim = 64
    num_vectors = 3
    batch_size = 2
    composer = EnhancedVectorComposer(dim)
    # Create test data
    vectors = [np.random.randn(dim) for _ in range(num_vectors)]
    state = np.random.randn(dim)
    # Test single composition
    composed, new_state = composer.compose(vectors, state)
    if composed.shape == (1, dim):
        composed = composed.squeeze(0)
    if new_state.shape == (1, dim):
        new_state = new_state.squeeze(0)
    assert composed.shape == (dim,)
    assert new_state.shape == (dim,)
    assert not np.isnan(composed).any()
    assert not np.isnan(new_state).any()
    # Test batch composition
    vector_batches = [[np.random.randn(dim) for _ in range(num_vectors)] 
                     for _ in range(batch_size)]
    states = np.random.randn(batch_size, dim)
    composed_batch, new_states = composer.batch_compose(vector_batches, states)
    assert composed_batch.shape == (batch_size, dim)
    assert new_states.shape == (batch_size, dim)
    assert not np.isnan(composed_batch).any()
    assert not np.isnan(new_states).any()

def test_temperature_effect():
    dim = 64
    num_vectors = 3
    
    composer = EnhancedVectorComposer(dim)
    vectors = [np.random.randn(dim) for _ in range(num_vectors)]
    state = np.random.randn(dim)
    
    # Test with different temperatures
    composed1, _ = composer.compose(vectors, state, temperature=0.0)
    composed2, _ = composer.compose(vectors, state, temperature=1.0)
    
    # Higher temperature should result in more variation
    assert not np.array_equal(composed1, composed2)

def test_device_handling():
    dim = 64
    num_vectors = 3
    # Test CPU
    composer_cpu = EnhancedVectorComposer(dim, device='cpu')
    vectors = [np.random.randn(dim) for _ in range(num_vectors)]
    state = np.random.randn(dim)
    composed_cpu, _ = composer_cpu.compose(vectors, state)
    if composed_cpu.shape == (1, dim):
        composed_cpu = composed_cpu.squeeze(0)
    assert composed_cpu.shape == (dim,)
    # Test CUDA if available
    if torch.cuda.is_available():
        composer_cuda = EnhancedVectorComposer(dim, device='cuda')
        composed_cuda, _ = composer_cuda.compose(vectors, state)
        if composed_cuda.shape == (1, dim):
            composed_cuda = composed_cuda.squeeze(0)
        assert composed_cuda.shape == (dim,) 