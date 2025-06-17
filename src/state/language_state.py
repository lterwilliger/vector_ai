import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import deque
import heapq

class VectorWrapper:
    """Wrapper for storing vectors with importance scores and semantic context."""
    
    def __init__(self, importance: float, vector: np.ndarray, semantic_context: Optional[np.ndarray] = None):
        self.importance = importance
        self.vector = vector  # Now stores quantized vector
        self.semantic_context = semantic_context
        self.last_accessed = 0
        self.x_min = 0.0  # Quantization parameters
        self.scale = 1.0
    
    def get_dequantized_vector(self) -> np.ndarray:
        """Get the dequantized vector."""
        return self.vector.astype(np.float32) * self.scale + self.x_min
    
    def __lt__(self, other):
        return self.importance < other.importance
    
    def __eq__(self, other):
        return (self.importance == other.importance and 
                np.array_equal(self.vector, other.vector) and
                (self.semantic_context is None and other.semantic_context is None or
                 np.array_equal(self.semantic_context, other.semantic_context)))

class VectorMemory:
    """Enhanced memory system for storing and retrieving vectors with semantic context."""
    
    def __init__(self, dim: int, max_size: int = 1000, context_dim: int = 64):
        self.dim = dim
        self.context_dim = context_dim
        self.max_size = max_size
        self.memory: List[VectorWrapper] = []
        self.vector_dict: Dict[str, VectorWrapper] = {}
        self.access_counter = 0
        
    def _vector_key(self, vector: np.ndarray) -> str:
        # Normalize and round for stable hashing
        norm_vec = vector / np.linalg.norm(vector)
        return str(np.round(norm_vec, 6).tobytes())
    
    def quantize_int8(self, x: np.ndarray) -> Tuple[np.ndarray, float, float]:
        x_min, x_max = x.min(), x.max()
        scale = (x_max - x_min) / 255 if x_max != x_min else 1.0
        q = np.round((x - x_min) / scale).astype(np.uint8)
        return q, x_min, scale
    
    def dequantize_int8(self, q: np.ndarray, x_min: float, scale: float) -> np.ndarray:
        return q.astype(np.float32) * scale + x_min
    
    def store(self, vector: np.ndarray, importance: float = 1.0, semantic_context: Optional[np.ndarray] = None) -> None:
        """Store a vector with its importance score and semantic context."""
        if len(self.memory) >= self.max_size:
            # Remove least important item
            removed = heapq.heappop(self.memory)
            del self.vector_dict[self._vector_key(removed.get_dequantized_vector())]
        
        # Quantize vector
        q_vector, x_min, scale = self.quantize_int8(vector)
        wrapper = VectorWrapper(importance, q_vector, semantic_context)
        wrapper.x_min = x_min
        wrapper.scale = scale
        wrapper.last_accessed = self.access_counter
        self.access_counter += 1
        
        # Add new item
        heapq.heappush(self.memory, wrapper)
        
        # Update dictionary
        vector_id = self._vector_key(vector)
        self.vector_dict[vector_id] = wrapper
    
    def retrieve(self, query: np.ndarray, semantic_context: Optional[np.ndarray] = None, k: int = 5) -> List[Tuple[np.ndarray, float]]:
        """Retrieve k most similar vectors to the query, considering semantic context."""
        if not self.memory:
            return []
        
        similarities = []
        for wrapper in self.memory:
            wrapper.last_accessed = self.access_counter
            self.access_counter += 1
            
            # Get dequantized vector for similarity computation
            deq_vector = wrapper.get_dequantized_vector()
            
            # Calculate vector similarity
            vector_sim = np.dot(query, deq_vector) / (np.linalg.norm(query) * np.linalg.norm(deq_vector))
            
            # Calculate context similarity if available
            context_sim = 1.0
            if semantic_context is not None and wrapper.semantic_context is not None:
                context_sim = np.dot(semantic_context, wrapper.semantic_context) / (
                    np.linalg.norm(semantic_context) * np.linalg.norm(wrapper.semantic_context)
                )
            
            # Combined similarity score
            combined_sim = vector_sim * 0.7 + context_sim * 0.3
            similarities.append((combined_sim, deq_vector, wrapper.importance))
        
        # Get top k
        similarities.sort(reverse=True)
        return [(vec, imp) for _, vec, imp in similarities[:k]]
    
    def update_importance(self, vector: np.ndarray, new_importance: float) -> None:
        """Update the importance score of a vector."""
        vector_id = self._vector_key(vector)
        if vector_id in self.vector_dict:
            wrapper = self.vector_dict[vector_id]
            # Remove old entry
            self.memory = [w for w in self.memory if w != wrapper]
            # Update importance and add back
            wrapper.importance = new_importance
            wrapper.last_accessed = self.access_counter
            self.access_counter += 1
            self.memory.append(wrapper)
            heapq.heapify(self.memory)

class StateTransitionNetwork(nn.Module):
    """Enhanced network for managing state transitions with semantic coherence."""
    
    def __init__(self, dim: int, hidden_dim: int = 512):
        super().__init__()
        self.dim = dim
        
        # Enhanced attention mechanism
        self.memory_attention = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)
        self.context_attention = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)
        
        # State transition network with residual connections
        self.transition_network = nn.Sequential(
            nn.Linear(dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, dim)
        )
        
        # Semantic coherence check
        self.coherence_check = nn.Sequential(
            nn.Linear(dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, 
                current_state: torch.Tensor,
                memory_vectors: torch.Tensor,
                input_vector: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Update state based on current state, memory, and input with semantic coherence check.
        
        Returns:
            Tuple of (updated_state, coherence_score)
        """
        # Handle single vector case
        is_single = current_state.dim() == 1
        if is_single:
            current_state = current_state.unsqueeze(0)
            if memory_vectors.dim() == 2:
                memory_vectors = memory_vectors.unsqueeze(0)
            if input_vector.dim() == 1:
                input_vector = input_vector.unsqueeze(0)
        
        # Ensure proper dimensions
        if current_state.dim() == 2:
            current_state = current_state.unsqueeze(1)
        if memory_vectors.dim() == 2:
            memory_vectors = memory_vectors.unsqueeze(0)
        if input_vector.dim() == 1:
            input_vector = input_vector.unsqueeze(0)
        
        # Attend to memory with context
        memory_attended, _ = self.memory_attention(
            current_state, memory_vectors, memory_vectors
        )
        
        # Context attention
        context_attended, _ = self.context_attention(
            current_state, input_vector.unsqueeze(1), input_vector.unsqueeze(1)
        )
        
        # Combine information
        current_state_flat = current_state.squeeze(1)
        memory_attended_flat = memory_attended.squeeze(1)
        context_attended_flat = context_attended.squeeze(1)
        
        combined = torch.cat([
            current_state_flat,
            memory_attended_flat,
            context_attended_flat
        ], dim=-1)
        
        # Generate new state
        new_state = self.transition_network(combined)
        
        # Check semantic coherence
        coherence_score = self.coherence_check(new_state).squeeze(-1)
        
        # Normalize
        new_state = new_state / torch.norm(new_state, dim=-1, keepdim=True)
        
        # Return to original shape if single vector
        if is_single:
            new_state = new_state.squeeze(0)
            coherence_score = coherence_score.squeeze(0)
            
        return new_state, coherence_score

class LanguageState:
    """Enhanced language state management with semantic coherence."""
    
    def __init__(self, 
                 dim: int,
                 memory_size: int = 1000,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.dim = dim
        self.device = device
        
        self.memory = VectorMemory(dim, memory_size)
        self.transition_network = StateTransitionNetwork(dim).to(device)
        self.current_state = torch.zeros(dim, device=device)
        self.semantic_context = torch.zeros(dim, device=device)
        
    def update(self, 
               vector: np.ndarray,
               importance: float = 1.0,
               set_importance_direct: bool = False) -> Tuple[np.ndarray, float]:
        """
        Update language state with new vector and semantic context.
        
        Returns:
            Tuple of (updated_state, coherence_score)
        """
        # Ensure vector is 1D and normalized
        if vector.ndim > 1:
            vector = vector.squeeze()
        vector = vector / np.linalg.norm(vector)
        
        # Convert to tensor
        vector_tensor = torch.from_numpy(vector).float().to(self.device)
        
        # Retrieve relevant memory with semantic context
        memory_vectors = self.memory.retrieve(
            vector, 
            semantic_context=self.semantic_context.cpu().numpy()
        )
        
        if memory_vectors:
            memory_tensor = torch.from_numpy(np.array([vec for vec, _ in memory_vectors])).float().to(self.device)
            # Normalize memory vectors
            memory_tensor = memory_tensor / torch.norm(memory_tensor, dim=-1, keepdim=True)
        else:
            memory_tensor = torch.zeros((1, self.dim), device=self.device)
        
        # Update state with coherence check
        with torch.no_grad():
            new_state, coherence_score = self.transition_network(
                self.current_state,
                memory_tensor,
                vector_tensor
            )
        
        # Update semantic context
        self.semantic_context = (self.semantic_context * 0.7 + new_state * 0.3).detach()
        self.semantic_context = self.semantic_context / torch.norm(self.semantic_context)
        
        # Store in memory with semantic context
        if set_importance_direct:
            store_importance = importance
        else:
            store_importance = importance * float(coherence_score)
        self.memory.store(
            vector, 
            importance=store_importance,
            semantic_context=self.semantic_context.cpu().numpy()
        )
        
        # Update current state
        self.current_state = new_state
        
        return self.current_state.cpu().numpy(), float(coherence_score)
    
    def get_state(self) -> np.ndarray:
        """Get current state vector."""
        return self.current_state.cpu().numpy()
    
    def get_semantic_context(self) -> np.ndarray:
        """Get current semantic context."""
        return self.semantic_context.cpu().numpy()
    
    def reset_state(self) -> None:
        """Reset state and semantic context to zero vectors."""
        self.current_state = torch.zeros(self.dim, device=self.device)
        self.semantic_context = torch.zeros(self.dim, device=self.device)
    
    def update_memory_importance(self, 
                               vector: np.ndarray,
                               new_importance: float) -> None:
        """Update importance score of a vector in memory."""
        self.memory.update_importance(vector, new_importance)
    
    def get_memory_vectors(self, k: int = 5) -> List[Tuple[np.ndarray, float]]:
        """Get k most important vectors from memory with their importance scores."""
        return [(wrapper.vector, wrapper.importance) 
                for wrapper in sorted(self.memory.memory, reverse=True)[:k]] 