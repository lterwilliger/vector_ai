import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Dict
from src.state.language_state import LanguageState
from src.operations.vector_operations import VectorOperations

class StateAwareGenerator(nn.Module):
    """Neural network for generating vectors while maintaining state awareness."""
    
    def __init__(self, dim: int, hidden_dim: int = 512):
        super().__init__()
        self.dim = dim
        
        # State processing
        self.state_encoder = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Memory processing
        self.memory_attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        
        # Generation network
        self.generator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # state + memory
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )
        
    def forward(self,
                state: torch.Tensor,
                memory_vectors: torch.Tensor) -> torch.Tensor:
        """
        Generate new vector based on state and memory.
        
        Args:
            state: Current state vector
            memory_vectors: Retrieved memory vectors
            
        Returns:
            Generated vector
        """
        # Ensure correct shapes: (batch, seq_len, dim)
        if state.dim() == 1:
            state = state.unsqueeze(0).unsqueeze(1)
        elif state.dim() == 2:
            state = state.unsqueeze(1)
        if memory_vectors.dim() == 2:
            memory_vectors = memory_vectors.unsqueeze(1)
        
        # Encode state
        state_encoded = self.state_encoder(state)
        memory_encoded = self.state_encoder(memory_vectors)
        memory_attended, _ = self.memory_attention(
            state_encoded, memory_encoded, memory_encoded
        )
        memory_attended = memory_attended.squeeze(1)
        
        # Combine state and memory
        combined = torch.cat([state_encoded.squeeze(1), memory_attended], dim=-1)
        
        # Generate new vector
        output = self.generator(combined)
        
        # Normalize
        return output / torch.norm(output, dim=-1, keepdim=True)

class StateAwareGenerationSystem:
    """Complete system for state-aware vector generation."""
    
    def __init__(self,
                 dim: int,
                 memory_size: int = 1000,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.dim = dim
        self.device = device
        
        # Initialize components
        self.state_manager = LanguageState(dim, memory_size, device)
        self.vector_ops = VectorOperations(dim)
        self.generator = StateAwareGenerator(dim).to(device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.generator.parameters())
        
    def generate(self,
                temperature: float = 1.0,
                num_samples: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate new vectors based on current state.
        
        Args:
            temperature: Controls randomness
            num_samples: Number of vectors to generate
            
        Returns:
            Tuple of (generated vectors, new states)
        """
        self.generator.eval()
        with torch.no_grad():
            # Get current state and memory
            current_state = torch.from_numpy(self.state_manager.get_state()).float().to(self.device)
            memory_vectors = self.state_manager.get_memory_vectors(k=5)
            
            if memory_vectors:
                memory_tensor = torch.from_numpy(np.array(memory_vectors)).float().to(self.device)
            else:
                memory_tensor = torch.zeros((1, self.dim), device=self.device)
            
            # Generate vectors
            generated_vectors = []
            new_states = []
            
            for _ in range(num_samples):
                # Generate base vector
                vector = self.generator(current_state.unsqueeze(0), memory_tensor.unsqueeze(0))
                vector = vector.squeeze(0)
                
                # Add temperature-based noise
                if temperature > 0:
                    noise = torch.randn_like(vector) * temperature
                    vector = vector + noise
                    vector = vector / torch.norm(vector, dim=-1, keepdim=True)
                
                # Update state
                new_state = self.state_manager.update(vector.cpu().numpy())
                
                generated_vectors.append(vector.cpu().numpy())
                new_states.append(new_state)
            
            return np.array(generated_vectors), np.array(new_states)
    
    def train_step(self,
                  target_vectors: np.ndarray,
                  importance: float = 1.0) -> float:
        """
        Perform one training step.
        
        Args:
            target_vectors: Target vectors to generate
            importance: Importance score for memory storage
            
        Returns:
            Loss value
        """
        self.generator.train()
        self.optimizer.zero_grad()
        
        # Convert to tensor
        target_tensor = torch.from_numpy(target_vectors).float().to(self.device)
        
        # Get current state and memory
        current_state = torch.from_numpy(self.state_manager.get_state()).float().to(self.device)
        memory_vectors = self.state_manager.get_memory_vectors(k=5)
        
        if memory_vectors:
            memory_tensor = torch.from_numpy(np.array(memory_vectors)).float().to(self.device)
        else:
            memory_tensor = torch.zeros((1, self.dim), device=self.device)
        
        # Generate vectors
        generated = self.generator(current_state.unsqueeze(0), memory_tensor.unsqueeze(0))
        
        # Calculate loss (cosine similarity loss)
        loss = 1 - torch.mean(torch.sum(generated * target_tensor, dim=1) / 
                            (torch.norm(generated, dim=1) * torch.norm(target_tensor, dim=1)))
        
        # Backpropagate
        loss.backward()
        self.optimizer.step()
        
        # Update state with generated vectors
        for vector in generated.detach().cpu().numpy():
            self.state_manager.update(vector, importance)
        
        return loss.item()
    
    def save_state(self, path: str) -> None:
        """Save the current state of the system."""
        state = {
            'generator_state': self.generator.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'state_manager': self.state_manager.get_state()
        }
        torch.save(state, path)
    
    def load_state(self, path: str) -> None:
        """Load a saved state."""
        state = torch.load(path)
        self.generator.load_state_dict(state['generator_state'])
        self.optimizer.load_state_dict(state['optimizer_state'])
        self.state_manager.reset_state()
        self.state_manager.update(state['state_manager']) 