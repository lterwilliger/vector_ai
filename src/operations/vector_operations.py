import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Union, Dict
from scipy.spatial.transform import Rotation
import math

class VectorOperations:
    """Operations that work directly in vector space without tokenization."""
    
    def __init__(self, dim: int):
        self.dim = dim
        
    def normalize(self, vector: np.ndarray) -> np.ndarray:
        """Normalize a vector to unit length."""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm
    
    def batch_normalize(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize a batch of vectors."""
        norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
        norms[norms == 0] = 1
        return vectors / norms
    
    def rotate(self, 
               vector: np.ndarray, 
               angle: float,
               axis: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Rotate a vector in 3D space. Only supports 3D vectors.
        """
        if self.dim != 3:
            raise ValueError("Rotation is only supported for 3D vectors.")
        if axis is None:
            # Generate random orthogonal axis
            axis = np.random.randn(self.dim)
            axis = axis - np.dot(axis, vector) * vector
            axis = self.normalize(axis)
        
        # Create rotation matrix
        rot = Rotation.from_rotvec(angle * axis)
        
        # Apply rotation
        return rot.apply(vector)
    
    def interpolate(self, 
                   vec1: np.ndarray,
                   vec2: np.ndarray,
                   alpha: float) -> np.ndarray:
        """
        Interpolate between two vectors in high-dimensional space.
        
        Args:
            vec1: First vector
            vec2: Second vector
            alpha: Interpolation factor (0 to 1)
        """
        # Ensure vectors are normalized
        vec1 = self.normalize(vec1)
        vec2 = self.normalize(vec2)
        
        # Spherical interpolation
        dot = np.dot(vec1, vec2)
        dot = np.clip(dot, -1.0, 1.0)
        omega = np.arccos(dot)
        
        if omega == 0:
            return vec1
        
        so = np.sin(omega)
        return np.sin((1.0 - alpha) * omega) / so * vec1 + np.sin(alpha * omega) / so * vec2
    
    def compose(self,
               vectors: List[np.ndarray],
               method: str = 'weighted_sum',
               weights: Optional[List[float]] = None) -> np.ndarray:
        """
        Compose multiple vectors using various methods.
        
        Args:
            vectors: List of vectors to compose
            method: Composition method ('weighted_sum', 'product', 'max', 'min')
            weights: Optional weights for weighted composition
        """
        if not vectors:
            raise ValueError("No vectors provided for composition")
        
        vectors = np.array(vectors)
        
        if method == 'weighted_sum':
            if weights is None:
                weights = [1.0] * len(vectors)
            weights = np.array(weights)
            weights = weights / np.sum(weights)
            result = np.sum(vectors * weights[:, np.newaxis], axis=0)
            
        elif method == 'product':
            result = np.prod(vectors, axis=0)
            
        elif method == 'max':
            result = np.max(vectors, axis=0)
            
        elif method == 'min':
            result = np.min(vectors, axis=0)
            
        else:
            raise ValueError(f"Unknown composition method: {method}")
        
        return self.normalize(result)
    
    def transform(self,
                 vector: np.ndarray,
                 operation: str,
                 **kwargs) -> np.ndarray:
        """
        Apply various transformations to a vector.
        
        Args:
            vector: Vector to transform
            operation: Transformation operation
            **kwargs: Additional arguments for the operation
        """
        if operation == 'negate':
            return -vector
            
        elif operation == 'intensify':
            factor = kwargs.get('factor', 1.5)
            return vector * factor
            
        elif operation == 'attenuate':
            factor = kwargs.get('factor', 0.5)
            return vector * factor
            
        elif operation == 'rotate':
            angle = kwargs.get('angle', math.pi/4)
            axis = kwargs.get('axis', None)
            return self.rotate(vector, angle, axis)
            
        elif operation == 'project':
            target = kwargs.get('target', None)
            if target is None:
                raise ValueError("Target vector required for projection")
            target = self.normalize(target)
            return np.dot(vector, target) * target
            
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def similarity(self,
                  vec1: np.ndarray,
                  vec2: np.ndarray,
                  metric: str = 'cosine') -> float:
        """
        Calculate similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            metric: Similarity metric ('cosine', 'euclidean', 'manhattan')
        """
        if metric == 'cosine':
            return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            
        elif metric == 'euclidean':
            return -np.linalg.norm(vec1 - vec2)
            
        elif metric == 'manhattan':
            return -np.sum(np.abs(vec1 - vec2))
            
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def batch_similarity(self,
                        vec1: np.ndarray,
                        vec2: np.ndarray,
                        metric: str = 'cosine') -> np.ndarray:
        """
        Calculate similarity between batches of vectors.
        
        Args:
            vec1: First batch of vectors
            vec2: Second batch of vectors
            metric: Similarity metric
        """
        if metric == 'cosine':
            # Normalize batches
            vec1 = self.batch_normalize(vec1)
            vec2 = self.batch_normalize(vec2)
            return np.sum(vec1 * vec2, axis=-1)
            
        elif metric == 'euclidean':
            return -np.linalg.norm(vec1 - vec2, axis=-1)
            
        elif metric == 'manhattan':
            return -np.sum(np.abs(vec1 - vec2), axis=-1)
            
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def cluster(self,
               vectors: np.ndarray,
               n_clusters: int,
               method: str = 'kmeans') -> Tuple[np.ndarray, np.ndarray]:
        """
        Cluster vectors in high-dimensional space.
        
        Args:
            vectors: Vectors to cluster
            n_clusters: Number of clusters
            method: Clustering method ('kmeans', 'spectral')
            
        Returns:
            Tuple of (cluster assignments, cluster centers)
        """
        from sklearn.cluster import KMeans, SpectralClustering
        
        if method == 'kmeans':
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            assignments = kmeans.fit_predict(vectors)
            centers = kmeans.cluster_centers_
            
        elif method == 'spectral':
            spectral = SpectralClustering(n_clusters=n_clusters, random_state=42)
            assignments = spectral.fit_predict(vectors)
            # Calculate cluster centers
            centers = np.array([np.mean(vectors[assignments == i], axis=0) 
                              for i in range(n_clusters)])
            
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        return assignments, centers 