from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from .base import VectorEmbedding

class SentenceTransformerEmbedding(VectorEmbedding):
    """Implementation of VectorEmbedding using sentence-transformers."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the sentence transformer embedding.
        
        Args:
            model_name: Name of the sentence-transformer model to use
        """
        self.model = SentenceTransformer(model_name)
        self.corpus = []  # Store corpus for vector-to-text conversion
        super().__init__(embedding_dim=self.model.get_sentence_embedding_dimension())
    
    def encode(self, text: str) -> np.ndarray:
        """Convert text into a continuous vector representation."""
        # Add text to corpus for later vector-to-text conversion
        if text not in self.corpus:
            self.corpus.append(text)
        return self.model.encode(text, convert_to_numpy=True)
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Convert multiple texts into vector representations."""
        # Add texts to corpus for later vector-to-text conversion
        for text in texts:
            if text not in self.corpus:
                self.corpus.append(text)
        return self.model.encode(texts, convert_to_numpy=True)
    
    def decode_vector(self, vector: np.ndarray) -> str:
        """
        Convert a vector back to text by finding the closest text in the corpus.
        
        Args:
            vector: The vector to decode
            
        Returns:
            The text that most closely matches the vector
        """
        if not self.corpus:
            return "No corpus available for vector-to-text conversion"
            
        # Encode the corpus if not already encoded
        corpus_embeddings = self.encode_batch(self.corpus)
        
        # Calculate similarities
        similarities = [
            self.similarity(vector, corpus_embedding)
            for corpus_embedding in corpus_embeddings
        ]
        
        # Get the most similar text
        best_idx = np.argmax(similarities)
        return self.corpus[best_idx]
    
    def decode(self, vector: np.ndarray) -> str:
        """
        Note: This is a placeholder as sentence-transformers don't provide direct decoding.
        In a real implementation, you might want to use a nearest-neighbor search
        against a corpus of texts to find the closest matching text.
        """
        raise NotImplementedError(
            "Direct decoding is not supported by sentence-transformers. "
            "Consider implementing a nearest-neighbor search against a corpus."
        )
    
    def semantic_search(self, query: str, corpus: List[str], top_k: int = 5) -> List[tuple]:
        """
        Perform semantic search over a corpus of texts.
        
        Args:
            query: The search query
            corpus: List of texts to search over
            top_k: Number of results to return
            
        Returns:
            List of (text, score) tuples
        """
        query_embedding = self.encode(query)
        corpus_embeddings = self.encode_batch(corpus)
        
        # Calculate similarities
        similarities = [
            self.similarity(query_embedding, corpus_embedding)
            for corpus_embedding in corpus_embeddings
        ]
        
        # Get top-k results
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(corpus[idx], similarities[idx]) for idx in top_indices] 