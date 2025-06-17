import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for vector composition."""
    
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        
        self.scale = self.head_dim ** -0.5
        
    def forward(self, vectors: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        batch_size = vectors.size(0)
        
        # Project queries, keys, and values
        q = self.query(state).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(vectors).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(vectors).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, 1, self.dim)
        
        return self.proj(out).squeeze(1)

class StateTransitionNetwork(nn.Module):
    """Network for managing state transitions in vector space."""
    
    def __init__(self, dim: int, hidden_dim: int = 512):
        super().__init__()
        self.dim = dim
        
        self.network = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )
        
    def forward(self, vector: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        # Concatenate vector and state
        combined = torch.cat([vector, state], dim=-1)
        # Generate new state
        new_state = self.network(combined)
        # Normalize
        return new_state / torch.norm(new_state, dim=-1, keepdim=True)

class EnhancedVectorComposer:
    """Enhanced vector composition with attention and state management."""
    
    def __init__(self, dim: int, num_heads: int = 8, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.dim = dim
        self.device = device
        
        self.attention = MultiHeadAttention(dim, num_heads).to(device)
        self.state_network = StateTransitionNetwork(dim).to(device)
        
    def compose(self, 
                vectors: List[np.ndarray], 
                state: np.ndarray,
                temperature: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compose vectors using attention and update state.
        
        Args:
            vectors: List of vectors to compose
            state: Current state vector
            temperature: Controls randomness in attention
            
        Returns:
            Tuple of (composed vector, new state)
        """
        # Convert to tensors
        vectors_tensor = torch.from_numpy(np.array(vectors)).float().to(self.device)
        state_tensor = torch.from_numpy(state).float().to(self.device)
        squeeze_output = False
        if len(vectors_tensor.shape) == 2:
            vectors_tensor = vectors_tensor.unsqueeze(0)
            squeeze_output = True
        if len(state_tensor.shape) == 1:
            state_tensor = state_tensor.unsqueeze(0)
            squeeze_output = True
        
        # Apply attention
        attended = self.attention(vectors_tensor, state_tensor)
        
        # Add temperature-based noise to attention
        if temperature > 0:
            noise = torch.randn_like(attended) * temperature
            attended = attended + noise
            attended = attended / torch.norm(attended, dim=-1, keepdim=True)
        
        # Update state
        new_state = self.state_network(attended, state_tensor)
        
        # Convert back to numpy
        attended_np = attended.detach().cpu().numpy()
        new_state_np = new_state.detach().cpu().numpy()
        if squeeze_output:
            attended_np = attended_np.squeeze(0)
            new_state_np = new_state_np.squeeze(0)
        return attended_np, new_state_np
    
    def batch_compose(self,
                     vector_batches: List[List[np.ndarray]],
                     states: np.ndarray,
                     temperature: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compose multiple batches of vectors.
        
        Args:
            vector_batches: List of vector batches
            states: Batch of state vectors
            temperature: Controls randomness in attention
            
        Returns:
            Tuple of (composed vectors, new states)
        """
        # Convert to tensors
        max_vectors = max(len(batch) for batch in vector_batches)
        padded_batches = []
        
        for batch in vector_batches:
            # Pad batch to max length
            padded = np.zeros((max_vectors, self.dim))
            padded[:len(batch)] = batch
            padded_batches.append(padded)
        
        vectors_tensor = torch.from_numpy(np.array(padded_batches)).float().to(self.device)
        states_tensor = torch.from_numpy(states).float().to(self.device)
        
        # Apply attention
        attended = self.attention(vectors_tensor, states_tensor)
        
        # Add temperature-based noise
        if temperature > 0:
            noise = torch.randn_like(attended) * temperature
            attended = attended + noise
            attended = attended / torch.norm(attended, dim=-1, keepdim=True)
        
        # Update states
        new_states = self.state_network(attended, states_tensor)
        
        return attended.detach().cpu().numpy(), new_states.detach().cpu().numpy() 