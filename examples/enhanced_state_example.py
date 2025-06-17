import torch
import numpy as np
from src.state.enhanced_state import EnhancedStateManager
from src.embeddings.sentence_transformer import SentenceTransformerEmbedding

def main():
    # Initialize embedding model
    embedding_model = SentenceTransformerEmbedding()
    vector_dim = 384  # Dimension of the embedding vectors
    
    # Initialize enhanced state manager
    state_manager = EnhancedStateManager(vector_dim)
    
    # Example texts for demonstration
    texts = [
        "The cat sat on the mat",
        "The dog ran in the park",
        "The bird flew in the sky",
        "The fish swam in the pond"
    ]
    
    # Convert texts to vectors
    vectors = [embedding_model.encode(text) for text in texts]
    
    print("\nDemonstrating state updates with navigation:")
    for i, (text, vector) in enumerate(zip(texts, vectors)):
        # Update state with navigation towards next vector
        target = vectors[(i + 1) % len(vectors)]
        new_state, coherence = state_manager.update(
            vector,
            target_vector=target,
            temperature=0.1
        )
        print(f"\nInput: {text}")
        print(f"Target: {texts[(i + 1) % len(texts)]}")
        print(f"Coherence score: {coherence:.4f}")
    
    print("\nDemonstrating interpolation:")
    # Interpolate between first and last vectors
    interpolated = state_manager.interpolate(
        vectors[0],
        vectors[-1],
        steps=5,
        temperature=0.1
    )
    print(f"Interpolated {len(interpolated)} vectors between:")
    print(f"Start: {texts[0]}")
    print(f"End: {texts[-1]}")
    
    print("\nDemonstrating extrapolation:")
    # Calculate direction vector
    direction = vectors[1] - vectors[0]
    direction = direction / np.linalg.norm(direction)
    
    # Extrapolate from last vector
    extrapolated = state_manager.extrapolate(
        vectors[-1],
        direction,
        steps=5,
        temperature=0.1
    )
    print(f"Extrapolated {len(extrapolated)} vectors from:")
    print(f"Start: {texts[-1]}")
    print(f"Direction: {texts[1]} -> {texts[0]}")
    
    print("\nDemonstrating training:")
    # Create some training data
    input_vectors = torch.from_numpy(np.array(vectors[:-1])).float()
    target_vectors = torch.from_numpy(np.array(vectors[1:])).float()
    
    # Train for a few steps
    for step in range(5):
        loss = state_manager.train_step(input_vectors, target_vectors)
        print(f"Training step {step + 1}, Loss: {loss:.4f}")

if __name__ == "__main__":
    main() 