import pytest
import numpy as np
import torch
from src.state.language_state import LanguageState, VectorMemory, StateTransitionNetwork

def test_vector_memory():
    dim = 64
    memory = VectorMemory(dim)
    
    # Test storing vectors
    vector1 = np.random.randn(dim)
    vector2 = np.random.randn(dim)
    memory.store(vector1, importance=1.0)
    memory.store(vector2, importance=2.0)
    
    # Test retrieving vectors
    retrieved = memory.retrieve(vector1)
    assert len(retrieved) > 0
    assert np.allclose(retrieved[0][0], vector1, atol=1e-5)  # Use higher tolerance for quantized vectors
    
    # Test importance update
    memory.update_importance(vector1, new_importance=3.0)
    retrieved = memory.retrieve(vector1)
    assert len(retrieved) > 0
    assert np.allclose(retrieved[0][0], vector1, atol=1e-5)

def test_state_transition_network():
    dim = 64
    batch_size = 2
    seq_len = 3
    network = StateTransitionNetwork(dim)
    
    # Create test tensors
    current_state = torch.randn(batch_size, dim)
    memory_vectors = torch.randn(batch_size, seq_len, dim)
    input_vector = torch.randn(batch_size, dim)
    
    # Test forward pass
    new_state, coherence_score = network(current_state, memory_vectors, input_vector)
    
    assert new_state.shape == (batch_size, dim)
    assert coherence_score.shape == (batch_size,)
    assert not torch.isnan(new_state).any()
    assert not torch.isinf(new_state).any()
    assert torch.all(coherence_score >= 0) and torch.all(coherence_score <= 1)
    
    # Test normalization
    norms = torch.norm(new_state, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

def test_language_state():
    dim = 64
    state = LanguageState(dim)
    
    # Test initial state
    initial_state = state.get_state()
    initial_context = state.get_semantic_context()
    assert initial_state.shape == (dim,)
    assert initial_context.shape == (dim,)
    assert np.allclose(initial_state, np.zeros(dim))
    assert np.allclose(initial_context, np.zeros(dim))
    
    # Test state update
    vector = np.random.randn(dim)
    new_state, coherence_score = state.update(vector)
    
    assert new_state.shape == (dim,)
    assert isinstance(coherence_score, float)
    assert 0 <= coherence_score <= 1
    assert not np.isnan(new_state).any()
    assert not np.isinf(new_state).any()
    
    # Test memory storage with semantic context
    memory_vectors = state.get_memory_vectors(k=1)
    assert len(memory_vectors) == 1
    assert isinstance(memory_vectors[0], tuple)
    assert memory_vectors[0][0].shape == (dim,)
    assert isinstance(memory_vectors[0][1], float)
    
    # Test importance update
    state.update_memory_importance(vector, new_importance=2.0)
    memory_vectors = state.get_memory_vectors(k=5)
    found = False
    norm_vector = vector / np.linalg.norm(vector)
    for vec, imp in memory_vectors:
        if np.allclose(vec, norm_vector, atol=1e-5):  # Use higher tolerance for quantized vectors
            assert imp == 2.0
            found = True
    assert found, "Updated vector not found in memory."
    
    # Test state reset
    state.reset_state()
    reset_state = state.get_state()
    reset_context = state.get_semantic_context()
    assert np.allclose(reset_state, np.zeros(dim))
    assert np.allclose(reset_context, np.zeros(dim))

def test_memory_retrieval():
    dim = 64
    memory = VectorMemory(dim)
    
    # Store some vectors
    vectors = [np.random.randn(dim) for _ in range(5)]
    for i, vec in enumerate(vectors):
        memory.store(vec, importance=float(i))
    
    # Test retrieval
    query = vectors[0]
    retrieved = memory.retrieve(query)
    assert len(retrieved) > 0
    assert np.allclose(retrieved[0][0], query, atol=1e-5)  # Use higher tolerance for quantized vectors

def test_device_handling():
    dim = 64
    
    # Test CPU
    state_cpu = LanguageState(dim, device='cpu')
    vector = np.random.randn(dim)
    new_state, coherence_score = state_cpu.update(vector)
    assert new_state.shape == (dim,)
    assert isinstance(coherence_score, float)
    
    # Test CUDA if available
    if torch.cuda.is_available():
        state_cuda = LanguageState(dim, device='cuda')
        new_state, coherence_score = state_cuda.update(vector)
        assert new_state.shape == (dim,)
        assert isinstance(coherence_score, float)

def test_semantic_coherence():
    dim = 64
    state = LanguageState(dim)
    
    # Test semantic coherence with related vectors
    vec1 = np.random.randn(dim)
    vec2 = np.random.randn(dim)
    vec3 = np.random.randn(dim)
    
    # Update with related vectors
    state1, score1 = state.update(vec1)
    state2, score2 = state.update(vec2)
    state3, score3 = state.update(vec3)
    
    # Check that coherence scores are reasonable
    assert all(0 <= score <= 1 for score in [score1, score2, score3])
    
    # Test semantic context evolution
    context = state.get_semantic_context()
    assert context.shape == (dim,)
    assert not np.allclose(context, np.zeros(dim))  # Context should have evolved 