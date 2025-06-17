from abc import ABC, abstractmethod
import numpy as np
import torch
from typing import Union, List, Dict, Any

class VectorEmbedding(ABC):
    """Base class for vector embeddings that operate in continuous space."""
    
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
    
    @abstractmethod
    def encode(self, text: str) -> np.ndarray:
        """Convert text into a continuous vector representation."""
        pass
    
    @abstractmethod
    def decode(self, vector: np.ndarray) -> str:
        """Convert a vector back into text representation."""
        pass
    
    def compose(self, vectors: List[np.ndarray], method: str = 'mean') -> np.ndarray:
        """Compose multiple vectors into a single vector representation.
        
        Args:
            vectors: List of vectors to compose
            method: Composition method ('mean', 'sum', 'max', 'min')
            
        Returns:
            Composed vector
        """
        if vectors is None or len(vectors) == 0:
            raise ValueError("No vectors provided for composition")
            
        vectors = np.array(vectors)
        
        if method == 'mean':
            return np.mean(vectors, axis=0)
        elif method == 'sum':
            return np.sum(vectors, axis=0)
        elif method == 'max':
            return np.max(vectors, axis=0)
        elif method == 'min':
            return np.min(vectors, axis=0)
        else:
            raise ValueError(f"Unknown composition method: {method}")
    
    def similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def to_tensor(self, vector: np.ndarray) -> torch.Tensor:
        """Convert numpy array to PyTorch tensor."""
        return torch.from_numpy(vector)
    
    def from_tensor(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert PyTorch tensor to numpy array."""
        return tensor.detach().cpu().numpy() 