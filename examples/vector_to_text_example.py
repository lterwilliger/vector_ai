import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import numpy as np
from src.embeddings.vector_to_text import VectorToTextPipeline
from src.embeddings.sentence_transformer import SentenceTransformerEmbedding

class TextPairDataset(Dataset):
    def __init__(self, file_path, max_pairs=None):
        self.pairs = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_pairs and i >= max_pairs:
                    break
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    input_text = parts[1]
                    target_text = parts[3]
                    self.pairs.append((input_text, target_text))
                else:
                    print(f"Skipping malformed line: {line.strip()}")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        return self.pairs[idx]

def main():
    # Initialize embedding model
    embedding_model = SentenceTransformerEmbedding()
    vector_dim = 384  # Dimension of the embedding vectors
    
    # Create dataset
    print("Loading dataset...")
    dataset = TextPairDataset('data/Sentence pairs in English-English - 2025-06-13.tsv', max_pairs=10000)
    print(f"Dataset size: {len(dataset)} pairs")
    
    # Create pipeline
    pipeline = VectorToTextPipeline(vector_dim, embedding_model)
    
    # Create data loader
    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training setup
    optimizer = torch.optim.Adam(pipeline.converter.parameters(), lr=1e-5)
    num_epochs = 10
    
    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (input_texts, target_texts) in enumerate(progress_bar):
            # Convert texts to vectors using embedding model
            input_vectors = torch.stack([torch.from_numpy(embedding_model.encode(text)).float() for text in input_texts])
            target_vectors = torch.stack([torch.from_numpy(embedding_model.encode(text)).float() for text in target_texts])
            
            # Ensure vectors are on the correct device
            input_vectors = input_vectors.to(pipeline.device)
            target_vectors = target_vectors.to(pipeline.device)
            
            # Training step
            loss = pipeline.train_step(input_vectors, target_vectors, optimizer)
            total_loss += loss
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss:.4f}'})
            
            # Save checkpoint every 100 batches
            if batch_idx % 100 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': pipeline.converter.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, f'checkpoints/checkpoint_epoch{epoch}_batch{batch_idx}.pt')
        
        avg_loss = total_loss / len(dataloader)
        print(f"\nEpoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}")
        
        # Save epoch checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': pipeline.converter.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, f'checkpoints/checkpoint_epoch{epoch}.pt')
    
    # Test generation
    print("\nTesting generation...")
    test_input = "The quick brown fox jumps over the lazy dog"
    test_vector = embedding_model.encode(test_input)
    generated_text = pipeline.convert_to_text(test_vector, temperature=0.7)
    print(f"Input: {test_input}")
    print(f"Generated: {generated_text}")

if __name__ == "__main__":
    main() 