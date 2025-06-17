import pytest
import numpy as np
import torch
import os
from src.composition.enhanced_composer import EnhancedVectorComposer
from src.state.language_state import LanguageState
from src.operations.vector_operations import VectorOperations
from src.generation.state_aware_generator import StateAwareGenerationSystem

class TestIntegratedSystem:
    """Test suite for the integrated vector-based language system."""
    
    @pytest.fixture
    def setup(self):
        """Set up test environment with all components."""
        self.dim = 64
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize all components
        self.vector_ops = VectorOperations(self.dim)
        self.state_manager = LanguageState(self.dim, device=self.device)
        self.composer = EnhancedVectorComposer(self.dim, device=self.device)
        self.generation_system = StateAwareGenerationSystem(self.dim, device=self.device)
        
        # Create some test vectors
        self.test_vectors = [np.random.randn(self.dim) for _ in range(5)]
        self.test_vectors = [self.vector_ops.normalize(vec) for vec in self.test_vectors]
    
    def test_vector_operations_integration(self, setup):
        """Test vector operations with other components."""
        # Test rotation and composition
        if self.dim != 3:
            # Skip rotation test for non-3D
            pass
        else:
            rotated = self.vector_ops.rotate(self.test_vectors[0], angle=np.pi/4)
            composed = self.vector_ops.compose([rotated, self.test_vectors[1]], method='weighted_sum')
            assert composed.shape == (self.dim,)
            assert np.isclose(np.linalg.norm(composed), 1.0)
        # Test interpolation with state
        state = self.state_manager.get_state()
        composed = self.vector_ops.compose([self.test_vectors[0], self.test_vectors[1]], method='weighted_sum')
        interpolated = self.vector_ops.interpolate(composed, state, alpha=0.5)
        assert interpolated.shape == (self.dim,)
        assert np.isclose(np.linalg.norm(interpolated), 1.0)
    
    def test_state_memory_integration(self, setup):
        """Test state and memory management with other components."""
        # Update state with vectors
        for vec in self.test_vectors:
            self.state_manager.update(vec, importance=1.0)
        
        # Test memory retrieval
        memory_vectors = self.state_manager.get_memory_vectors(k=3)
        assert len(memory_vectors) == 3
        
        # Test memory importance update
        self.state_manager.update_memory_importance(memory_vectors[0], importance=2.0)
        updated_memory = self.state_manager.get_memory_vectors(k=1)
        assert len(updated_memory) == 1
    
    def test_composition_generation_integration(self, setup):
        """Test vector composition with generation."""
        # Compose vectors
        composed, new_state = self.composer.compose(self.test_vectors[:3], self.state_manager.get_state())
        if composed.shape[0] == 1:
            composed = composed.squeeze(0)
        if new_state.shape[0] == 1:
            new_state = new_state.squeeze(0)
        assert composed.shape == (self.dim,)
        assert new_state.shape == (self.dim,)
        # Use composed vector for generation
        generated, states = self.generation_system.generate(temperature=0.5, num_samples=2)
        assert generated.shape == (2, self.dim)
        assert states.shape == (2, self.dim)
    
    def test_end_to_end_generation(self, setup):
        """Test complete generation pipeline."""
        # Initialize state with some vectors
        for vec in self.test_vectors[:2]:
            self.state_manager.update(vec, importance=1.0)
        
        # Generate new vectors
        generated, states = self.generation_system.generate(temperature=0.7, num_samples=3)
        
        # Compose generated vectors
        composed, new_state = self.composer.compose(generated, states[-1])
        
        # Update state with composed vector
        final_state = self.state_manager.update(composed, importance=1.0)
        
        assert composed.shape == (self.dim,)
        assert final_state.shape == (self.dim,)
        assert np.isclose(np.linalg.norm(composed), 1.0)
        assert np.isclose(np.linalg.norm(final_state), 1.0)
    
    def test_training_integration(self, setup):
        """Test training with all components."""
        # Generate target vectors
        target_vectors = np.array([self.vector_ops.normalize(np.random.randn(self.dim)) 
                                 for _ in range(2)])
        
        # Train on targets
        loss = self.generation_system.train_step(target_vectors)
        
        assert isinstance(loss, float)
        assert not np.isnan(loss)
        
        # Generate new vectors after training
        generated, states = self.generation_system.generate(temperature=0.5)
        
        assert generated.shape == (1, self.dim)
        assert states.shape == (1, self.dim)
    
    def test_state_persistence_integration(self, setup):
        """Test state persistence across all components."""
        # Generate and update state
        generated, states = self.generation_system.generate(num_samples=2)
        for vec in generated:
            self.state_manager.update(vec, importance=1.0)
        
        # Save state
        save_path = "test_integrated_state.pt"
        self.generation_system.save_state(save_path)
        
        # Create new system and load state
        new_system = StateAwareGenerationSystem(self.dim, device=self.device)
        new_system.load_state(save_path)
        
        # Compare states
        original_state = self.state_manager.get_state()
        loaded_state = new_system.state_manager.get_state()
        
        assert np.allclose(original_state, loaded_state)
        
        # Clean up
        os.remove(save_path)
    
    def test_memory_aware_generation(self, setup):
        """Test generation with memory awareness."""
        # Store important vectors in memory
        for i, vec in enumerate(self.test_vectors):
            self.state_manager.update(vec, importance=float(i + 1))
        
        # Generate with different temperatures
        vectors1, _ = self.generation_system.generate(temperature=0.0)
        vectors2, _ = self.generation_system.generate(temperature=1.0)
        
        assert not np.array_equal(vectors1, vectors2)
        
        # Test memory influence
        memory_vectors = self.state_manager.get_memory_vectors(k=2)
        assert len(memory_vectors) == 2
    
    def test_vector_transformations(self, setup):
        """Test vector transformations with state."""
        # Get current state
        state = self.state_manager.get_state()
        
        # Apply various transformations
        transformed = self.vector_ops.transform(state, 'intensify', factor=1.5)
        assert np.linalg.norm(transformed) > np.linalg.norm(state)
        
        transformed = self.vector_ops.transform(state, 'attenuate', factor=0.5)
        assert np.linalg.norm(transformed) < np.linalg.norm(state)
        
        transformed = self.vector_ops.transform(state, 'rotate', angle=np.pi/4)
        assert np.isclose(np.linalg.norm(transformed), np.linalg.norm(state))
    
    def test_clustering_integration(self, setup):
        """Test vector clustering with generation."""
        # Generate multiple vectors
        generated, _ = self.generation_system.generate(num_samples=10)
        
        # Cluster generated vectors
        assignments, centers = self.vector_ops.cluster(generated, n_clusters=3)
        
        assert assignments.shape == (10,)
        assert centers.shape == (3, self.dim)
        assert set(assignments) == set(range(3))
        
        # Use cluster centers for generation
        for center in centers:
            self.state_manager.update(center, importance=1.0)
        
        new_generated, _ = self.generation_system.generate(temperature=0.5)
        assert new_generated.shape == (1, self.dim)
    
    def test_similarity_metrics(self, setup):
        """Test similarity metrics across components."""
        # Generate vectors
        generated, _ = self.generation_system.generate(num_samples=2)
        
        # Test different similarity metrics
        cosine_sim = self.vector_ops.similarity(generated[0], generated[1], metric='cosine')
        euclidean_sim = self.vector_ops.similarity(generated[0], generated[1], metric='euclidean')
        manhattan_sim = self.vector_ops.similarity(generated[0], generated[1], metric='manhattan')
        
        assert isinstance(cosine_sim, float)
        assert isinstance(euclidean_sim, float)
        assert isinstance(manhattan_sim, float)
        
        # Test batch similarity
        batch_sim = self.vector_ops.batch_similarity(generated, generated, metric='cosine')
        assert batch_sim.shape == (2,) 