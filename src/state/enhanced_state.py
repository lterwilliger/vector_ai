import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import deque
import heapq
from .language_state import VectorMemory, VectorWrapper

class LatentSpaceNavigator(nn.Module):
    """Enhanced navigation through latent space with semantic awareness."""
    
    def __init__(self, dim: int, hidden_dim: int = 512):
        super().__init__()
        self.dim = dim
        
        # Navigation network
        self.navigation_net = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )
        
        # Semantic coherence network
        self.coherence_net = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, 
                current_vector: torch.Tensor,
                target_vector: torch.Tensor,
                temperature: float = 1.0) -> Tuple[torch.Tensor, float]:
        """
        Navigate from current vector towards target vector.
        
        Args:
            current_vector: Current position in latent space
            target_vector: Target position in latent space
            temperature: Controls exploration (higher = more random)
            
        Returns:
            Tuple of (new_vector, coherence_score)
        """
        # Combine current and target vectors
        combined = torch.cat([current_vector, target_vector], dim=-1)
        
        # Generate navigation vector
        navigation = self.navigation_net(combined)
        
        # Add temperature-based noise
        if temperature > 0:
            noise = torch.randn_like(navigation) * temperature
            navigation = navigation + noise
        
        # Normalize navigation vector
        navigation = navigation / torch.norm(navigation, dim=-1, keepdim=True)
        
        # Calculate semantic coherence
        coherence = self.coherence_net(combined)
        
        return navigation, coherence

class EnhancedStateManager:
    """Enhanced state management with improved vector space navigation."""
    
    def __init__(self, 
                 dim: int,
                 memory_size: int = 1000,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.dim = dim
        self.device = device
        
        # Initialize components
        self.memory = VectorMemory(dim, memory_size)
        self.navigator = LatentSpaceNavigator(dim).to(device)
        self.current_state = torch.zeros(dim, device=device)
        self.state_history = deque(maxlen=100)  # Track state history
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.navigator.parameters())
    
    def update(self, 
               vector: np.ndarray,
               target_vector: Optional[np.ndarray] = None,
               temperature: float = 1.0,
               importance: float = 1.0) -> Tuple[np.ndarray, float]:
        """
        Update state with new vector and optionally navigate towards target.
        
        Args:
            vector: New vector to incorporate
            target_vector: Optional target vector to navigate towards
            temperature: Controls exploration in navigation
            importance: Importance score for memory storage
            
        Returns:
            Tuple of (updated_state, coherence_score)
        """
        # Convert to tensor
        vector_tensor = torch.from_numpy(vector).float().to(self.device)
        
        if target_vector is not None:
            target_tensor = torch.from_numpy(target_vector).float().to(self.device)
            # Navigate towards target
            new_state, coherence = self.navigator(
                self.current_state.unsqueeze(0),
                target_tensor.unsqueeze(0),
                temperature
            )
            new_state = new_state.squeeze(0)
        else:
            # Simple state update
            new_state = vector_tensor
            coherence = torch.tensor(1.0, device=self.device)
        
        # Store current state in history before updating
        self.state_history.append(vector)  # Store the input vector directly
        
        # Update current state
        self.current_state = new_state.detach()  # Detach to prevent gradient tracking
        
        # Store in memory (detach before converting to numpy)
        self.memory.store(new_state.detach().cpu().numpy(), importance)
        
        return new_state.detach().cpu().numpy(), coherence.item()
    
    def interpolate(self, 
                   start_vector: np.ndarray,
                   end_vector: np.ndarray,
                   steps: int = 10,
                   temperature: float = 0.1) -> List[np.ndarray]:
        """
        Generate interpolated path between two vectors.
        
        Args:
            start_vector: Starting vector
            end_vector: Ending vector
            steps: Number of interpolation steps
            temperature: Controls exploration in interpolation
            
        Returns:
            List of interpolated vectors
        """
        vectors = []
        current = torch.from_numpy(start_vector).float().to(self.device)
        target = torch.from_numpy(end_vector).float().to(self.device)
        
        for _ in range(steps):
            # Navigate towards target
            new_vector, _ = self.navigator(
                current.unsqueeze(0),
                target.unsqueeze(0),
                temperature
            )
            new_vector = new_vector.squeeze(0)
            
            # Store vector (detach before converting to numpy)
            vectors.append(new_vector.detach().cpu().numpy())
            current = new_vector.detach()  # Detach to prevent gradient tracking
        
        return vectors
    
    def extrapolate(self,
                   vector: np.ndarray,
                   direction: np.ndarray,
                   steps: int = 10,
                   temperature: float = 0.1) -> List[np.ndarray]:
        """
        Generate extrapolated path from vector in given direction.
        
        Args:
            vector: Starting vector
            direction: Direction vector for extrapolation
            steps: Number of extrapolation steps
            temperature: Controls exploration in extrapolation
            
        Returns:
            List of extrapolated vectors
        """
        vectors = []
        current = torch.from_numpy(vector).float().to(self.device)
        direction = torch.from_numpy(direction).float().to(self.device)
        direction = direction / torch.norm(direction)
        
        for _ in range(steps):
            # Calculate target by moving in direction
            target = current + direction
            
            # Navigate towards target
            new_vector, _ = self.navigator(
                current.unsqueeze(0),
                target.unsqueeze(0),
                temperature
            )
            new_vector = new_vector.squeeze(0)
            
            # Normalize the new vector
            new_vector = new_vector / torch.norm(new_vector)
            
            # Store vector (detach before converting to numpy)
            vectors.append(new_vector.detach().cpu().numpy())
            current = new_vector.detach()  # Detach to prevent gradient tracking
        
        return vectors
    
    def train_step(self,
                  input_vectors: torch.Tensor,
                  target_vectors: torch.Tensor) -> float:
        """
        Perform one training step for the navigator.
        
        Args:
            input_vectors: Input vectors
            target_vectors: Target vectors
            
        Returns:
            Loss value
        """
        self.navigator.train()
        self.optimizer.zero_grad()
        
        # Generate navigation vectors
        navigation_vectors, coherence = self.navigator(input_vectors, target_vectors)
        
        # Calculate navigation loss (cosine similarity)
        nav_loss = 1 - torch.mean(torch.sum(navigation_vectors * target_vectors, dim=1) / 
                                (torch.norm(navigation_vectors, dim=1) * torch.norm(target_vectors, dim=1)))
        
        # Calculate coherence loss
        coh_loss = torch.mean((coherence - 1.0) ** 2)
        
        # Combined loss
        loss = nav_loss + 0.1 * coh_loss
        
        # Backpropagate
        loss.backward()
        self.optimizer.step()
        
        return loss.item() 