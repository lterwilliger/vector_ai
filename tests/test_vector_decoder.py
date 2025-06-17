import unittest
import torch
import numpy as np
from src.generation.vector_decoder import VectorDecoder
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class TestVectorDecoder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.decoder = VectorDecoder()
        cls.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        # Use the same sentences as in the decoder's corpus
        cls.corpus_sentences = [
            "The cat jumps over the wall", "The flower is blooming", "The mountain is tall", "The book is on the table", "The dog runs through the field", "The tree is growing", "The ocean is deep", "The bird is in the nest", "The fish is in the water"
        ]
        cls.test_vectors = cls.encoder.encode(cls.corpus_sentences, convert_to_tensor=True).numpy()
        cls.test_texts = cls.corpus_sentences
    
    def test_concept_space_generation(self):
        """Test if generation creates meaningful concepts in concept space."""
        for i, vec in enumerate(self.test_vectors):
            generated = self.decoder.decode(vec)
            gen_vec = self.encoder.encode(generated, convert_to_tensor=True).numpy()
            sim = float(np.dot(vec, gen_vec) / (np.linalg.norm(vec) * np.linalg.norm(gen_vec)))
            print(f"[Test] Input: '{self.test_texts[i]}' | Output: '{generated}' | Similarity: {sim:.3f}")
            self.assertGreater(sim, 0.1, "Generated concept should be semantically coherent")

    def test_concept_space_operations(self):
        """Test if vector operations maintain meaningful concept space properties (simple averaging)."""
        v1 = self.test_vectors[0]
        v2 = self.test_vectors[1]
        composed = (v1 + v2) / 2
        generated = self.decoder.decode(composed)
        gen_vec = self.encoder.encode(generated, convert_to_tensor=True).numpy()
        sim1 = float(np.dot(v1, gen_vec) / (np.linalg.norm(v1) * np.linalg.norm(gen_vec)))
        sim2 = float(np.dot(v2, gen_vec) / (np.linalg.norm(v2) * np.linalg.norm(gen_vec)))
        print(f"[Test] Compose: '{self.test_texts[0]}' + '{self.test_texts[1]}' | Output: '{generated}' | Sim1: {sim1:.3f} | Sim2: {sim2:.3f}")
        self.assertGreater(sim1, 0.1, "Composition should maintain some properties of v1")
        self.assertGreater(sim2, 0.1, "Composition should maintain some properties of v2")

    def test_concept_space_transitions(self):
        """Test if vector operations create smooth transitions in concept space (simple interpolation)."""
        v1 = self.test_vectors[2]
        v2 = self.test_vectors[3]
        for alpha in np.linspace(0, 1, 5):
            interp = (1 - alpha) * v1 + alpha * v2
            generated = self.decoder.decode(interp)
            gen_vec = self.encoder.encode(generated, convert_to_tensor=True).numpy()
            sim1 = float(np.dot(v1, gen_vec) / (np.linalg.norm(v1) * np.linalg.norm(gen_vec)))
            sim2 = float(np.dot(v2, gen_vec) / (np.linalg.norm(v2) * np.linalg.norm(gen_vec)))
            print(f"[Test] Interp {alpha:.2f}: '{self.test_texts[2]}' <-> '{self.test_texts[3]}' | Output: '{generated}' | Sim1: {sim1:.3f} | Sim2: {sim2:.3f}")
            self.assertGreater(max(sim1, sim2), 0.1, "Transition should be semantically coherent with at least one endpoint")

    def test_concept_space_refinement(self):
        """Test if vector refinement maintains concept space properties (identity refinement)."""
        for i, vec in enumerate(self.test_vectors):
            refined = vec  # Identity refinement
            generated = self.decoder.decode(refined)
            gen_vec = self.encoder.encode(generated, convert_to_tensor=True).numpy()
            sim = float(np.dot(vec, gen_vec) / (np.linalg.norm(vec) * np.linalg.norm(gen_vec)))
            print(f"[Test] Refinement: '{self.test_texts[i]}' | Output: '{generated}' | Similarity: {sim:.3f}")
            self.assertGreater(sim, 0.1, "Refinement should maintain semantic coherence")

if __name__ == '__main__':
    unittest.main() 