import torch
import numpy as np
from src.state.enhanced_state import EnhancedStateManager
from src.embeddings.sentence_transformer import SentenceTransformerEmbedding
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

def evaluate_semantic_progression():
    """Evaluate semantic progression through vector space."""
    # Initialize components
    embedding_model = SentenceTransformerEmbedding()
    state_manager = EnhancedStateManager(384)
    
    # Create semantic progression
    progression = [
        "A small seed",
        "A growing plant",
        "A young tree",
        "A mature tree",
        "A forest"
    ]
    
    # Convert to vectors
    vectors = [embedding_model.encode(text) for text in progression]
    
    # Track states and coherences
    states = []
    coherences = []
    similarities = []
    
    # Update state through progression
    for i, (text, vector) in enumerate(zip(progression, vectors)):
        target = vectors[i+1] if i < len(vectors)-1 else None
        new_state, coherence = state_manager.update(
            vector,
            target_vector=target,
            temperature=0.1
        )
        states.append(new_state)
        coherences.append(coherence)
        
        # Calculate similarity with next vector
        if i < len(vectors)-1:
            sim = cosine_similarity([new_state], [vectors[i+1]])[0][0]
            similarities.append(sim)
    
    # Print results
    print("\nSemantic Progression Evaluation:")
    print("-" * 50)
    for i, (text, coherence) in enumerate(zip(progression, coherences)):
        print(f"Step {i+1}: {text}")
        print(f"Coherence: {coherence:.4f}")
        if i < len(similarities):
            print(f"Similarity to next: {similarities[i]:.4f}")
        print()
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(coherences, 'b-o', label='Coherence')
    plt.title('Semantic Coherence')
    plt.xlabel('Step')
    plt.ylabel('Coherence Score')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(similarities, 'r-o', label='Similarity')
    plt.title('Similarity to Next Vector')
    plt.xlabel('Step')
    plt.ylabel('Cosine Similarity')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('semantic_progression.png')
    plt.close()

def evaluate_interpolation():
    """Evaluate vector interpolation."""
    # Initialize components
    embedding_model = SentenceTransformerEmbedding()
    state_manager = EnhancedStateManager(384)
    
    # Create pairs for interpolation
    pairs = [
        ("A cat", "A dog"),
        ("Hot water", "Cold water"),
        ("Day", "Night"),
        ("Happy", "Sad")
    ]
    
    print("\nInterpolation Evaluation:")
    print("-" * 50)
    
    for start, end in pairs:
        # Convert to vectors
        start_vec = embedding_model.encode(start)
        end_vec = embedding_model.encode(end)
        
        # Interpolate
        interpolated = state_manager.interpolate(
            start_vec,
            end_vec,
            steps=5
        )
        
        # Calculate similarities
        similarities = []
        for vec in interpolated:
            sim_start = cosine_similarity([vec], [start_vec])[0][0]
            sim_end = cosine_similarity([vec], [end_vec])[0][0]
            similarities.append((sim_start, sim_end))
        
        # Print results
        print(f"\nInterpolating from '{start}' to '{end}':")
        for i, (sim_start, sim_end) in enumerate(similarities):
            print(f"Step {i+1}:")
            print(f"  Similarity to start: {sim_start:.4f}")
            print(f"  Similarity to end: {sim_end:.4f}")
        
        # Plot similarities
        plt.figure(figsize=(8, 4))
        steps = range(len(similarities))
        plt.plot(steps, [s[0] for s in similarities], 'b-o', label='Start')
        plt.plot(steps, [s[1] for s in similarities], 'r-o', label='End')
        plt.title(f'Interpolation: {start} → {end}')
        plt.xlabel('Step')
        plt.ylabel('Cosine Similarity')
        plt.legend()
        plt.savefig(f'interpolation_{start.replace(" ", "_")}_{end.replace(" ", "_")}.png')
        plt.close()

def evaluate_extrapolation():
    """Evaluate vector extrapolation."""
    # Initialize components
    embedding_model = SentenceTransformerEmbedding()
    state_manager = EnhancedStateManager(384)
    
    # Create sequences for extrapolation
    sequences = [
        ["Small", "Medium", "Large"],
        ["Cold", "Warm", "Hot"],
        ["Slow", "Medium", "Fast"],
        ["Quiet", "Moderate", "Loud"]
    ]
    
    print("\nExtrapolation Evaluation:")
    print("-" * 50)
    
    for sequence in sequences:
        # Convert to vectors
        vectors = [embedding_model.encode(text) for text in sequence]
        
        # Calculate direction
        direction = vectors[1] - vectors[0]
        direction = direction / np.linalg.norm(direction)
        
        # Extrapolate from last vector
        extrapolated = state_manager.extrapolate(
            vectors[-1],
            direction,
            steps=5
        )
        
        # Calculate similarities with sequence
        similarities = []
        for vec in extrapolated:
            sims = [cosine_similarity([vec], [v])[0][0] for v in vectors]
            similarities.append(sims)
        
        # Print results
        print(f"\nExtrapolating from sequence: {' → '.join(sequence)}")
        for i, sims in enumerate(similarities):
            print(f"Step {i+1}:")
            for j, sim in enumerate(sims):
                print(f"  Similarity to {sequence[j]}: {sim:.4f}")
        
        # Plot similarities
        plt.figure(figsize=(10, 4))
        steps = range(len(similarities))
        for j, word in enumerate(sequence):
            plt.plot(steps, [s[j] for s in similarities], 'o-', label=word)
        plt.title(f'Extrapolation: {" → ".join(sequence)}')
        plt.xlabel('Step')
        plt.ylabel('Cosine Similarity')
        plt.legend()
        plt.savefig(f'extrapolation_{"_".join(sequence)}.png')
        plt.close()

def main():
    print("Evaluating Enhanced State Management System")
    print("=" * 50)
    
    # Run evaluations
    evaluate_semantic_progression()
    evaluate_interpolation()
    evaluate_extrapolation()
    
    print("\nEvaluation complete. Check the generated plots for visual results.")

if __name__ == "__main__":
    main() 