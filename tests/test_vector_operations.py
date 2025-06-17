import pytest
import numpy as np
from src.operations.vector_operations import VectorOperations

def test_normalize():
    dim = 64
    ops = VectorOperations(dim)
    
    # Test single vector normalization
    vector = np.random.randn(dim)
    normalized = ops.normalize(vector)
    
    assert normalized.shape == (dim,)
    assert np.isclose(np.linalg.norm(normalized), 1.0)
    
    # Test batch normalization
    vectors = np.random.randn(10, dim)
    normalized_batch = ops.batch_normalize(vectors)
    
    assert normalized_batch.shape == (10, dim)
    assert np.allclose(np.linalg.norm(normalized_batch, axis=1), 1.0)

def test_rotate():
    dim = 64
    ops = VectorOperations(dim)
    if dim != 3:
        pytest.skip("Rotation only supported for 3D vectors.")
    # Test rotation
    vector = np.random.randn(dim)
    vector = ops.normalize(vector)
    # Rotate by 90 degrees
    rotated = ops.rotate(vector, angle=np.pi/2)
    assert rotated.shape == (dim,)
    assert np.isclose(np.linalg.norm(rotated), 1.0)
    # Test that rotation preserves length
    assert np.isclose(np.linalg.norm(rotated), np.linalg.norm(vector))

def test_interpolate():
    dim = 64
    ops = VectorOperations(dim)
    
    # Create two vectors
    vec1 = np.random.randn(dim)
    vec2 = np.random.randn(dim)
    
    # Test interpolation
    alpha = 0.5
    interpolated = ops.interpolate(vec1, vec2, alpha)
    
    assert interpolated.shape == (dim,)
    assert np.isclose(np.linalg.norm(interpolated), 1.0)
    
    # Test endpoints
    start = ops.interpolate(vec1, vec2, 0.0)
    end = ops.interpolate(vec1, vec2, 1.0)
    
    assert np.allclose(start, ops.normalize(vec1))
    assert np.allclose(end, ops.normalize(vec2))

def test_compose():
    dim = 64
    ops = VectorOperations(dim)
    
    # Create test vectors
    vectors = [np.random.randn(dim) for _ in range(3)]
    weights = [0.5, 0.3, 0.2]
    
    # Test different composition methods
    weighted_sum = ops.compose(vectors, method='weighted_sum', weights=weights)
    product = ops.compose(vectors, method='product')
    max_comp = ops.compose(vectors, method='max')
    min_comp = ops.compose(vectors, method='min')
    
    assert weighted_sum.shape == (dim,)
    assert product.shape == (dim,)
    assert max_comp.shape == (dim,)
    assert min_comp.shape == (dim,)
    
    # Test normalization
    assert np.isclose(np.linalg.norm(weighted_sum), 1.0)
    assert np.isclose(np.linalg.norm(product), 1.0)
    assert np.isclose(np.linalg.norm(max_comp), 1.0)
    assert np.isclose(np.linalg.norm(min_comp), 1.0)

def test_transform():
    dim = 64
    ops = VectorOperations(dim)
    vector = np.random.randn(dim)
    vector = ops.normalize(vector)
    # Test different transformations
    negated = ops.transform(vector, 'negate')
    intensified = ops.transform(vector, 'intensify', factor=2.0)
    attenuated = ops.transform(vector, 'attenuate', factor=0.5)
    if dim == 3:
        rotated = ops.transform(vector, 'rotate', angle=np.pi/4)
        assert rotated.shape == (dim,)
    assert negated.shape == (dim,)
    assert intensified.shape == (dim,)
    assert attenuated.shape == (dim,)
    # Test that negation inverts the vector
    assert np.allclose(negated, -vector)
    # Test that intensification increases magnitude
    assert np.linalg.norm(intensified) > np.linalg.norm(vector)
    # Test that attenuation decreases magnitude
    assert np.linalg.norm(attenuated) < np.linalg.norm(vector)

def test_similarity():
    dim = 64
    ops = VectorOperations(dim)
    
    # Create test vectors
    vec1 = np.random.randn(dim)
    vec2 = np.random.randn(dim)
    
    # Test different similarity metrics
    cosine_sim = ops.similarity(vec1, vec2, metric='cosine')
    euclidean_sim = ops.similarity(vec1, vec2, metric='euclidean')
    manhattan_sim = ops.similarity(vec1, vec2, metric='manhattan')
    
    assert isinstance(cosine_sim, float)
    assert isinstance(euclidean_sim, float)
    assert isinstance(manhattan_sim, float)
    
    # Test batch similarity
    batch1 = np.random.randn(5, dim)
    batch2 = np.random.randn(5, dim)
    
    batch_cosine = ops.batch_similarity(batch1, batch2, metric='cosine')
    assert batch_cosine.shape == (5,)

def test_cluster():
    dim = 64
    ops = VectorOperations(dim)
    
    # Create test vectors
    vectors = np.random.randn(100, dim)
    
    # Test different clustering methods
    kmeans_assignments, kmeans_centers = ops.cluster(vectors, n_clusters=3, method='kmeans')
    spectral_assignments, spectral_centers = ops.cluster(vectors, n_clusters=3, method='spectral')
    
    assert kmeans_assignments.shape == (100,)
    assert kmeans_centers.shape == (3, dim)
    assert spectral_assignments.shape == (100,)
    assert spectral_centers.shape == (3, dim)
    
    # Test that assignments are valid
    assert set(kmeans_assignments) == set(range(3))
    assert set(spectral_assignments) == set(range(3)) 