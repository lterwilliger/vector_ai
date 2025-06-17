from src.embeddings.sentence_transformer import SentenceTransformerEmbedding

def main():
    # Initialize the embedding model
    embedder = SentenceTransformerEmbedding()
    
    # Example texts
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "A fast orange fox leaps across a sleeping canine",
        "The weather is beautiful today",
        "I love programming and artificial intelligence"
    ]
    
    # Encode texts into vectors
    print("Encoding texts into vectors...")
    vectors = embedder.encode_batch(texts)
    print(f"Vector shape: {vectors.shape}")
    
    # Demonstrate semantic composition
    print("\nDemonstrating semantic composition...")
    composed_vector = embedder.compose(vectors, method='mean')
    print(f"Composed vector shape: {composed_vector.shape}")
    
    # Perform semantic search
    print("\nPerforming semantic search...")
    query = "Tell me about the fox"
    results = embedder.semantic_search(query, texts, top_k=2)
    
    print(f"\nQuery: {query}")
    print("Top matches:")
    for text, score in results:
        print(f"Score: {score:.4f} - {text}")

if __name__ == "__main__":
    main() 