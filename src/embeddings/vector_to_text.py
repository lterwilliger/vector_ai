import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple, Dict
from .base import VectorEmbedding
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    _HAS_NLTK = True
except ImportError:
    _HAS_NLTK = False

class TextPairDataset(Dataset):
    """Dataset for (input_text, target_text) pairs."""
    def __init__(self, pairs: List[Tuple[str, str]], embedding_model: VectorEmbedding, vocabulary):
        self.pairs = pairs
        self.embedding_model = embedding_model
        self.vocabulary = vocabulary

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        input_text, target_text = self.pairs[idx]
        input_vector = self.embedding_model.encode(input_text)
        target_vectors = self.vocabulary.encode_text(target_text)
        return (
            torch.from_numpy(input_vector).float(),
            torch.from_numpy(target_vectors).float(),
            input_text,
            target_text
        )

def collate_batch(batch):
    # Pad target sequences to the max length in the batch
    input_vectors, target_vectors, input_texts, target_texts = zip(*batch)
    input_vectors = torch.stack(input_vectors)
    lengths = [t.shape[0] for t in target_vectors]
    max_len = max(lengths)
    padded_targets = torch.zeros(len(batch), max_len, target_vectors[0].shape[1])
    for i, t in enumerate(target_vectors):
        padded_targets[i, :t.shape[0], :] = t
    return input_vectors, padded_targets, lengths, input_texts, target_texts

class Vocabulary:
    """Vocabulary for mapping between tokens and vectors."""
    def __init__(self, tokens: List[str], embedding_model: VectorEmbedding, eos_token: str = "<EOS>"):
        self.tokens = tokens
        self.embedding_model = embedding_model
        self.eos_token = eos_token
        # Add EOS token if not present
        if eos_token not in tokens:
            self.tokens.append(eos_token)
        # Precompute token vectors
        self.token_vectors = np.stack([embedding_model.encode(token) for token in self.tokens])
        # Normalize for cosine similarity
        self.token_vectors = self.token_vectors / np.linalg.norm(self.token_vectors, axis=1, keepdims=True)
        self.token_to_idx = {token: i for i, token in enumerate(self.tokens)}
        self.idx_to_token = {i: token for i, token in enumerate(self.tokens)}

    def decode_vector(self, vector: np.ndarray) -> str:
        """Find the closest token to the given vector using cosine similarity."""
        vector = vector / np.linalg.norm(vector)
        sims = np.dot(self.token_vectors, vector)
        idx = np.argmax(sims)
        return self.idx_to_token[idx]

    def decode_sequence(self, sequence: np.ndarray) -> str:
        """Decode a sequence of vectors to a string of tokens."""
        tokens = [self.decode_vector(vec) for vec in sequence]
        # Optionally, collapse repeated tokens
        collapsed = [tokens[0]]
        for t in tokens[1:]:
            if t != collapsed[-1]:
                collapsed.append(t)
        return ' '.join(collapsed)

    def encode_text(self, text: str) -> np.ndarray:
        """Encode text to a sequence of token indices."""
        # Simple tokenization (split by space)
        tokens = text.split()
        # Add EOS token
        tokens.append(self.eos_token)
        # Convert tokens to indices
        indices = [self.token_to_idx.get(token, 0) for token in tokens]  # Use 0 as unknown token index
        return np.array(indices)

    def indices_to_vectors(self, indices: np.ndarray) -> np.ndarray:
        """Convert token indices to their corresponding vectors."""
        return self.token_vectors[indices]

class VectorToTextConverter(nn.Module):
    """Neural network for converting vectors back to text representations."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, verbose: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.verbose = verbose
        
        # Encoder: MLP to process input vectors
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Decoder: GRU to generate output vectors
        self.decoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Layer normalization for hidden state
        self.hidden_norm = nn.LayerNorm(hidden_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, max_length: int = 100) -> torch.Tensor:
        """
        Forward pass to convert input vectors to output vectors
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            max_length: Maximum sequence length to generate
            
        Returns:
            Output tensor of shape [batch_size, max_length, input_dim]
        """
        batch_size = x.size(0)
        encoded = self.encoder(x)  # [batch_size, hidden_dim]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
        outputs = []
        current_input = encoded.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        for _ in range(max_length):
            h0 = self.hidden_norm(h0)
            output, h0 = self.decoder(current_input, h0)
            outputs.append(output)  # [batch_size, 1, hidden_dim]
            current_input = output  # Use GRU output directly as next input
            
        outputs = torch.cat(outputs, dim=1)  # [batch_size, max_length, hidden_dim]
        final_output = self.output_proj(outputs)  # [batch_size, max_length, input_dim]
        return final_output
    
    def train_step(self, input_vectors, target_vectors, optimizer, mask=None):
        """
        Train the model on a batch of data.
        Args:
            input_vectors: Input vectors [batch_size, input_dim]
            target_vectors: Target vectors [batch_size, input_dim]
            optimizer: PyTorch optimizer
            mask: Optional mask for valid positions [batch_size, seq_length]
        Returns:
            float: Loss value
        """
        self.train()
        optimizer.zero_grad()
        
        # Forward pass - generate sequence of vectors
        output_vectors = self(input_vectors, max_length=1)  # [batch_size, 1, input_dim]
        output_vectors = output_vectors.squeeze(1)  # [batch_size, input_dim]
        
        # Ensure target vectors have the same shape
        if len(target_vectors.shape) == 3:
            target_vectors = target_vectors.squeeze(1)  # [batch_size, input_dim]
        
        # Cosine similarity loss with epsilon for stability
        eps = 1e-8
        dot = torch.sum(output_vectors * target_vectors, dim=-1)
        norm1 = torch.norm(output_vectors, dim=-1)
        norm2 = torch.norm(target_vectors, dim=-1)
        cosine = dot / (norm1 * norm2 + eps)
        loss = 1 - cosine
        
        # Mask out padded positions if mask is provided
        if mask is not None:
            loss = loss * mask
            loss = loss.sum() / (mask.sum() + eps)
        else:
            loss = loss.mean()
            
        # Add L2 regularization
        l2_lambda = 0.01
        l2_reg = torch.tensor(0., device=input_vectors.device)
        for param in self.parameters():
            l2_reg += torch.norm(param)
        loss += l2_lambda * l2_reg
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        optimizer.step()
        
        return loss.item()

class VectorToTextPipeline:
    """Pipeline for converting vectors to text using a learned mapping."""
    
    def __init__(self, vector_dim: int, embedding_model: nn.Module, verbose: bool = False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(embedding_model, nn.Module):
            self.embedding_model = embedding_model.to(self.device)
        else:
            self.embedding_model = embedding_model
        self.converter = VectorToTextConverter(vector_dim, hidden_dim=256, verbose=verbose).to(self.device)
        self.verbose = verbose
        
    def train_step(self,
                  input_vectors: torch.Tensor,
                  target_vectors: torch.Tensor,
                  optimizer: torch.optim.Optimizer) -> float:
        """
        Perform one training step.
        
        Args:
            input_vectors: Input vectors [batch_size, vector_dim]
            target_vectors: Target vectors [batch_size, vector_dim]
            optimizer: Optimizer to use
            
        Returns:
            Loss value
        """
        return self.converter.train_step(input_vectors, target_vectors, optimizer)
    
    def convert_to_text(self,
                       vector: np.ndarray,
                       max_length: int = 100,
                       temperature: float = 0.7) -> str:
        """
        Convert a vector to text by generating a sequence of vectors and finding the closest text.
        
        Args:
            vector: Input vector
            max_length: Maximum sequence length to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        self.converter.eval()
        with torch.no_grad():
            # Convert input to tensor
            x = torch.from_numpy(vector).float().to(self.device)
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            
            # Generate sequence of vectors
            sequence = self.converter(x, max_length)
            
            # Convert each vector in the sequence to text using the embedding model
            texts = []
            for vec in sequence[0].cpu().numpy():
                # Find the closest text representation using the embedding model
                text = self.embedding_model.decode_vector(vec)
                texts.append(text)
            
            return ' '.join(texts)

    def train_on_text(self, input_text: str, target_text: str, optimizer: torch.optim.Optimizer) -> float:
        """
        Train the converter on real text data.
        
        Args:
            input_text: Input text to encode
            target_text: Target text to encode
            optimizer: Optimizer to use
            
        Returns:
            Loss value
        """
        if self.embedding_model is None:
            raise ValueError("Embedding model must be provided for real data training.")
        
        # Encode input text to vector
        input_vector = self.embedding_model.encode(input_text)
        input_vector = torch.from_numpy(input_vector).float().to(self.device).unsqueeze(0)
        
        # Encode target text to sequence of vectors
        target_vectors = self.embedding_model.encode_text(target_text)
        target_vectors = torch.from_numpy(target_vectors).float().to(self.device).unsqueeze(0)
        
        # Perform training step
        return self.train_step(input_vector, target_vectors, optimizer)

    def evaluate(self, input_text: str, target_text: str) -> float:
        """
        Evaluate the converter on real text data using token-level accuracy and BLEU (if available).
        """
        if self.embedding_model is None:
            raise ValueError("Embedding model must be provided for evaluation.")
        input_vector = self.embedding_model.encode(input_text)
        generated_text = self.convert_to_text(input_vector)
        # Tokenize and ignore <EOS>
        gen_tokens = [t for t in generated_text.split() if t != self.embedding_model.eos_token]
        tgt_tokens = [t for t in target_text.split() if t != self.embedding_model.eos_token]
        # Token-level accuracy
        min_len = min(len(gen_tokens), len(tgt_tokens))
        correct = sum(g == t for g, t in zip(gen_tokens, tgt_tokens))
        accuracy = correct / max(len(tgt_tokens), 1)
        # BLEU score (optional)
        bleu = None
        if _HAS_NLTK:
            smoothie = SmoothingFunction().method4
            bleu = sentence_bleu([tgt_tokens], gen_tokens, smoothing_function=smoothie)
        return accuracy if bleu is None else (accuracy, bleu)

    def batch_train_on_texts(self, dataloader, optimizer, epochs=1, verbose=True):
        for epoch in range(epochs):
            epoch_loss = 0.0
            for input_vectors, target_vectors, lengths, _, _ in dataloader:
                input_vectors = input_vectors.to(self.device)
                target_vectors = target_vectors.to(self.device)
                # Optionally, mask loss for padded positions
                loss = self.train_step(input_vectors, target_vectors, optimizer)
                epoch_loss += loss
            if verbose:
                print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(dataloader):.4f}")

    def batch_evaluate(self, dataloader):
        total_acc = 0.0
        total_bleu = 0.0
        n = 0
        for input_vectors, target_vectors, lengths, input_texts, target_texts in dataloader:
            for i in range(len(input_texts)):
                acc_bleu = self.evaluate(input_texts[i], target_texts[i])
                if isinstance(acc_bleu, tuple):
                    acc, bleu = acc_bleu
                    total_bleu += bleu
                else:
                    acc = acc_bleu
                total_acc += acc
                n += 1
        avg_acc = total_acc / n
        avg_bleu = total_bleu / n if _HAS_NLTK else None
        return avg_acc if avg_bleu is None else (avg_acc, avg_bleu) 