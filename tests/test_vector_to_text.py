import pytest
import torch
import numpy as np
from src.embeddings.vector_to_text import VectorToTextConverter, VectorToTextPipeline
from src.embeddings.base import VectorEmbedding

class MockEmbedding(VectorEmbedding):
    """Mock embedding model for testing."""
    
    def __init__(self, embedding_dim: int):
        super().__init__(embedding_dim)
        
    def encode(self, text: str) -> np.ndarray:
        # Return a random normalized vector
        vec = np.random.randn(self.embedding_dim)
        return vec / np.linalg.norm(vec)
    
    def decode(self, vector: np.ndarray) -> str:
        # Mock implementation
        return "Mock decoded text"

@pytest.fixture
def vector_dim():
    return 384

@pytest.fixture
def batch_size():
    return 4

@pytest.fixture
def seq_length():
    return 10

@pytest.fixture
def converter(vector_dim):
    return VectorToTextConverter(vector_dim)

@pytest.fixture
def pipeline(vector_dim):
    embedding_model = MockEmbedding(vector_dim)
    return VectorToTextPipeline(vector_dim, embedding_model)

def test_converter_initialization(vector_dim):
    """Test converter initialization."""
    converter = VectorToTextConverter(vector_dim)
    assert converter.vector_dim == vector_dim
    assert converter.hidden_dim == 512  # default value
    
    # Test with custom parameters
    converter = VectorToTextConverter(
        vector_dim=256,
        hidden_dim=1024,
        num_layers=4,
        dropout=0.2
    )
    assert converter.vector_dim == 256
    assert converter.hidden_dim == 1024

def test_converter_forward(converter, batch_size, vector_dim, seq_length):
    """Test forward pass of converter."""
    # Create input vector
    x = torch.randn(batch_size, vector_dim)
    
    # Test forward pass
    output = converter(x, max_length=seq_length)
    
    # Check output shape
    assert output.shape == (batch_size, seq_length, vector_dim)
    
    # Check normalization
    norms = torch.norm(output, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

def test_converter_temperature(converter, batch_size, vector_dim):
    """Test temperature parameter in forward pass."""
    x = torch.randn(batch_size, vector_dim)
    
    # Test with different temperatures
    output_zero = converter(x, temperature=0.0)
    output_high = converter(x, temperature=1.0)
    
    # Higher temperature should produce more diverse outputs
    assert not torch.allclose(output_zero, output_high)

def test_pipeline_initialization(vector_dim):
    """Test pipeline initialization."""
    embedding_model = MockEmbedding(vector_dim)
    pipeline = VectorToTextPipeline(vector_dim, embedding_model)
    
    assert pipeline.vector_dim == vector_dim
    assert pipeline.embedding_model == embedding_model
    assert isinstance(pipeline.converter, VectorToTextConverter)

def test_pipeline_train_step(pipeline, batch_size, vector_dim, seq_length):
    """Test training step of pipeline."""
    # Create input and target vectors
    input_vectors = torch.randn(batch_size, vector_dim)
    target_vectors = torch.randn(batch_size, seq_length, vector_dim)
    
    # Normalize target vectors
    target_vectors = target_vectors / torch.norm(target_vectors, dim=-1, keepdim=True)
    
    # Create optimizer
    optimizer = torch.optim.Adam(pipeline.converter.parameters())
    
    # Test training step
    loss = pipeline.train_step(input_vectors, target_vectors, optimizer)
    
    assert isinstance(loss, float)
    assert loss >= 0

def test_pipeline_convert_to_text(pipeline, vector_dim):
    """Test text conversion."""
    # Create input vector
    vector = np.random.randn(vector_dim)
    vector = vector / np.linalg.norm(vector)
    
    # Test conversion
    text = pipeline.convert_to_text(vector)
    
    assert isinstance(text, str)
    assert len(text) > 0

def test_pipeline_device_handling(vector_dim):
    """Test device handling in pipeline."""
    # Test CPU
    embedding_model = MockEmbedding(vector_dim)
    pipeline_cpu = VectorToTextPipeline(vector_dim, embedding_model, device='cpu')
    assert pipeline_cpu.device == 'cpu'
    
    # Test CUDA if available
    if torch.cuda.is_available():
        pipeline_cuda = VectorToTextPipeline(vector_dim, embedding_model, device='cuda')
        assert pipeline_cuda.device == 'cuda'

def test_converter_gradient_flow(converter, batch_size, vector_dim):
    """Test gradient flow through converter."""
    x = torch.randn(batch_size, vector_dim, requires_grad=True)
    output = converter(x)
    
    # Test backward pass
    loss = output.mean()
    loss.backward()
    
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()

def test_pipeline_batch_handling(pipeline, batch_size, vector_dim):
    """Test batch handling in pipeline."""
    # Create batch of vectors
    vectors = np.random.randn(batch_size, vector_dim)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    
    # Test single vector
    text_single = pipeline.convert_to_text(vectors[0])
    assert isinstance(text_single, str)
    
    # Test batch of vectors
    texts_batch = [pipeline.convert_to_text(vec) for vec in vectors]
    assert len(texts_batch) == batch_size
    assert all(isinstance(text, str) for text in texts_batch) 