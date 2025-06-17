import torch
import torch.optim as optim
import numpy as np
from src.embeddings.sentence_transformer import SentenceTransformerEmbedding
from src.generation.vector_generator import VectorGenerator
from src.generation.vector_decoder import VectorDecoder

def create_training_data(embedder: SentenceTransformerEmbedding) -> tuple:
    """Create a diverse training dataset of related sentences."""
    # Create pairs of related sentences with different styles and variations
    sentence_pairs = [
        # Basic variations
        ("The cat sat on the mat", "A feline rested on the carpet"),
        ("The sun is bright today", "The weather is sunny"),
        ("I love programming", "Coding is my passion"),
        ("The movie was amazing", "The film was incredible"),
        ("She is very happy", "She feels joyful"),
        ("The book is interesting", "The novel is engaging"),
        ("He runs fast", "He is a quick runner"),
        ("The food tastes good", "The meal is delicious"),
        
        # Action variations
        ("The dog is running", "The canine sprints across the field"),
        ("Birds fly in the sky", "Avians soar through the air"),
        ("Fish swim in the ocean", "Marine creatures glide through the water"),
        ("The car drives on the road", "The vehicle travels along the highway"),
        
        # Emotion variations
        ("She is very sad", "She feels deeply sorrowful"),
        ("He is extremely angry", "He is filled with rage"),
        ("They are excited", "They feel thrilled"),
        ("The child is scared", "The young one feels frightened"),
        
        # Description variations
        ("The mountain is tall", "The peak reaches high into the sky"),
        ("The river is wide", "The waterway spans a great distance"),
        ("The forest is dense", "The woodland is thick with trees"),
        ("The city is busy", "The urban area bustles with activity"),
        
        # Abstract concepts
        ("Time passes quickly", "The moments fly by"),
        ("Life is beautiful", "Existence is wonderful"),
        ("Knowledge is power", "Understanding brings strength"),
        ("Change is constant", "Transformation is ongoing")
    ]
    
    # Convert to vectors
    input_texts = [pair[0] for pair in sentence_pairs]
    target_texts = [pair[1] for pair in sentence_pairs]
    
    input_vectors = embedder.encode_batch(input_texts)
    target_vectors = embedder.encode_batch(target_texts)
    
    return input_vectors, target_vectors, input_texts, target_texts

def main():
    # Initialize the embedding model
    embedder = SentenceTransformerEmbedding()
    
    # Create training data
    input_vectors, target_vectors, input_texts, target_texts = create_training_data(embedder)
    
    # Convert to tensors
    input_tensors = torch.from_numpy(input_vectors).float()
    target_tensors = torch.from_numpy(target_vectors).float()
    
    # Initialize the generator
    generator = VectorGenerator(
        input_dim=input_vectors.shape[1],
        output_dim=target_vectors.shape[1]
    )
    
    # Initialize the decoder
    decoder = VectorDecoder()
    
    # Initialize optimizer with lower learning rate
    optimizer = optim.Adam(generator.parameters(), lr=0.0001)  # Even lower learning rate
    
    # Training loop with more epochs
    num_epochs = 300  # More epochs for better convergence
    for epoch in range(num_epochs):
        loss = generator.train_step(input_tensors, target_tensors, optimizer)
        
        if (epoch + 1) % 30 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")
    
    # Test generation with multiple examples
    test_sentences = [
        "The dog is running",
        "She is very happy",
        "The mountain is tall",
        "Time passes quickly"
    ]
    
    print("\nTesting generation...")
    for test_sentence in test_sentences:
        print(f"\nInput: {test_sentence}")
        
        # Generate vector with very low temperature for more focused generation
        test_vector = embedder.encode(test_sentence)
        generated_vector = generator.generate(test_vector, temperature=0.1)  # Much lower temperature
        
        # Decode to text with very controlled parameters
        generated_text = decoder.decode(
            generated_vector,
            temperature=0.5,  # Lower temperature
            top_k=20,  # Fewer tokens to consider
            top_p=0.7,  # More focused sampling
            repetition_penalty=2.0  # Strong repetition penalty
        )
        print(f"Generated text: {generated_text}")
        
        # Find closest training target
        similarities = [
            embedder.similarity(generated_vector, target_vectors[i])
            for i in range(len(target_vectors))
        ]
        best_idx = np.argmax(similarities)
        print(f"Most similar training target: {target_texts[best_idx]}")
        print(f"Similarity score: {float(similarities[best_idx]):.4f}")

if __name__ == "__main__":
    main() 