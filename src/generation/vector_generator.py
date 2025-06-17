import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional

class VectorGenerator(nn.Module):
    """A neural network that generates new vectors in semantic space."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 512, output_dim: int = 384):
        """
        Initialize the vector generator.
        
        Args:
            input_dim: Dimension of input vectors
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output vectors
        """
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate a new vector from input vector."""
        return self.network(x)
    
    def generate(self, 
                input_vector: np.ndarray, 
                temperature: float = 1.0,
                num_samples: int = 1) -> np.ndarray:
        """
        Generate new vectors from an input vector.
        
        Args:
            input_vector: Input vector to generate from
            temperature: Controls randomness (higher = more random)
            num_samples: Number of vectors to generate
            
        Returns:
            Generated vectors
        """
        self.eval()
        with torch.no_grad():
            # Convert input to tensor
            x = torch.from_numpy(input_vector).float()
            if len(x.shape) == 1:
                x = x.unsqueeze(0)  # Add batch dimension
            
            # Generate base output
            output = self.forward(x)
            
            # Add noise based on temperature
            if temperature > 0:
                noise = torch.randn_like(output) * temperature
                output = output + noise
            
            # Normalize the output
            output = output / torch.norm(output, dim=1, keepdim=True)
            
            # Convert back to numpy
            return output.numpy()
    
    def train_step(self, 
                  input_vectors: torch.Tensor, 
                  target_vectors: torch.Tensor,
                  optimizer: torch.optim.Optimizer) -> float:
        """
        Perform one training step.
        
        Args:
            input_vectors: Input vectors
            target_vectors: Target vectors to generate
            optimizer: Optimizer to use
            
        Returns:
            Loss value
        """
        self.train()
        optimizer.zero_grad()
        
        # Generate vectors
        output = self.forward(input_vectors)
        
        # Calculate loss (cosine similarity loss)
        loss = 1 - torch.mean(torch.sum(output * target_vectors, dim=1) / 
                            (torch.norm(output, dim=1) * torch.norm(target_vectors, dim=1)))
        
        # Backpropagate
        loss.backward()
        optimizer.step()
        
        return loss.item() 