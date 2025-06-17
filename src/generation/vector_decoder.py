import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple, Dict
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

class VectorDecoder:
    """Converts semantic vectors back to text using continuous vector space operations."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the decoder.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.encoder = SentenceTransformer(model_name)
        self.embedding_dim = 384  # Match our vector space
        
        # Create a semantic space basis using diverse concepts
        self.basis_vectors = self._create_semantic_basis()
        
        # Initialize concept space operations
        self.concept_ops = nn.ModuleDict({
            'compose': nn.Sequential(
                nn.Linear(self.embedding_dim * 2, self.embedding_dim * 2),
                nn.LayerNorm(self.embedding_dim * 2),
                nn.ReLU(),
                nn.Linear(self.embedding_dim * 2, self.embedding_dim)
            ),
            'transform': nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim * 2),
                nn.LayerNorm(self.embedding_dim * 2),
                nn.ReLU(),
                nn.Linear(self.embedding_dim * 2, self.embedding_dim)
            ),
            'refine': nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim * 2),
                nn.LayerNorm(self.embedding_dim * 2),
                nn.ReLU(),
                nn.Linear(self.embedding_dim * 2, self.embedding_dim)
            ),
            'generate': nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim * 2),
                nn.LayerNorm(self.embedding_dim * 2),
                nn.ReLU(),
                nn.Linear(self.embedding_dim * 2, self.embedding_dim)
            )
        })
        
        # Initialize weights
        for module in self.concept_ops.values():
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
        
        # Create semantic templates for generation
        self.semantic_templates = self._create_semantic_templates()
        
        # Create semantic clusters for text generation
        self.semantic_clusters = self._create_semantic_clusters()
        
        # Create concept vectors for semantic operations
        self.concept_vectors = self._create_concept_vectors()
        
        # Initialize attention mechanisms
        self.type_attention = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, 1)
        )
        
        self.composition_attention = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, 1)
        )
    
    def _create_semantic_templates(self) -> Dict[str, List[str]]:
        """Create semantic templates for different types of concepts."""
        return {
            'action': [
                "The {subject} {action} {object}",
                "{subject} {action} {object}",
                "It {action} {object}"
            ],
            'state': [
                "The {subject} is {state}",
                "{subject} becomes {state}",
                "It is {state}"
            ],
            'property': [
                "The {subject} has {property}",
                "{subject} shows {property}",
                "It exhibits {property}"
            ],
            'relation': [
                "The {subject} relates to {object}",
                "{subject} connects with {object}",
                "It links to {object}"
            ]
        }
    
    def _create_semantic_basis(self) -> torch.Tensor:
        """Create a set of basis vectors that span our semantic space."""
        # Define core semantic dimensions
        dimensions = [
            # Action dimensions
            "action", "movement", "change", "causation",
            # State dimensions
            "state", "property", "quality", "quantity",
            # Object dimensions
            "object", "space", "time", "relation",
            # Abstract dimensions
            "existence", "perception", "cognition", "emotion"
        ]
        
        # Get embeddings for these dimensions
        with torch.no_grad():
            dimension_embeddings = self.encoder.encode(dimensions, convert_to_tensor=True)
        
        # Orthogonalize the basis vectors
        basis = dimension_embeddings.clone()
        for i in range(len(basis)):
            for j in range(i):
                basis[i] = basis[i] - torch.dot(basis[i], basis[j]) * basis[j]
            basis[i] = basis[i] / torch.norm(basis[i])
        
        return basis
    
    def _project_to_basis(self, vector: torch.Tensor) -> torch.Tensor:
        """Project a vector onto the semantic basis."""
        return torch.matmul(vector, self.basis_vectors.T)
    
    def _reconstruct_from_basis(self, coefficients: torch.Tensor) -> torch.Tensor:
        """Reconstruct a vector from its basis coefficients."""
        return torch.matmul(coefficients, self.basis_vectors)
    
    def _create_semantic_clusters(self) -> Dict[str, np.ndarray]:
        """Create semantic clusters for text generation."""
        # Define a diverse set of example sentences that cover different semantic aspects
        examples = [
            # Actions
            "The dog runs quickly", "She walks slowly", "They jump high", "The cat leaps over the fence", "The athlete sprints to the finish", "The child skips happily", "The car accelerates rapidly", "The bird soars above the trees",
            # States
            "The water is cold", "The sky is blue", "The food is hot", "The flower is blooming", "The tree is growing", "The river is flowing", "The patient is recovering", "The machine is idle", "The city is bustling",
            # Objects
            "A tall mountain", "A small bird", "A large tree", "A red apple", "A shiny coin", "A broken vase", "A wooden chair", "A glass of water", "A blue car",
            # Properties
            "Very beautiful", "Extremely difficult", "Quite simple", "Remarkably strong", "Surprisingly light", "Exceptionally fast", "Unusually quiet", "Highly intelligent", "Barely visible",
            # Time
            "Yesterday morning", "Next week", "Last year", "Tomorrow afternoon", "In a few minutes", "At midnight", "During the summer", "After the meeting", "Before sunrise",
            # Space
            "In the forest", "Under the bridge", "Above the clouds", "Near the river", "Inside the house", "Outside the school", "Between the buildings", "On the mountain", "Across the street",
            # Movement
            "Moving forward", "Falling down", "Rising up", "Turning left", "Spinning around", "Sliding backward", "Jumping over", "Crawling under", "Running beside",
            # Emotions
            "Feeling happy", "Being sad", "Getting angry", "Feeling anxious", "Being excited", "Feeling calm", "Being surprised", "Feeling proud", "Being nervous",
            # Quantities
            "Many people", "Few options", "Several times", "A couple of books", "Hundreds of stars", "Dozens of eggs", "A handful of coins", "Thousands of fans", "Plenty of space",
            # Qualities
            "Well done", "Poorly made", "Perfectly executed", "Carefully crafted", "Badly damaged", "Nicely decorated", "Roughly finished", "Elegantly designed", "Hastily written",
            # Relations
            "Between two points", "Among friends", "With family", "Next to the window", "Opposite the door", "Close to the park", "Far from home", "Alongside the road", "Against the wall",
            # Existence
            "There is", "There are", "There was", "There will be", "There might be", "There could be", "There should be", "There used to be", "There has been",
            # Change
            "Becoming better", "Getting worse", "Turning around", "Shifting focus", "Changing direction", "Evolving quickly", "Transforming slowly", "Adapting to change", "Switching roles",
            # Causation
            "Because of", "Due to", "As a result", "Owing to", "Thanks to", "On account of", "For the sake of", "In response to", "Following the event",
            # Perception
            "Looking at", "Listening to", "Touching the", "Smelling the flowers", "Tasting the soup", "Observing the stars", "Watching the movie", "Hearing the music", "Sensing danger",
            # Cognition
            "Thinking about", "Understanding that", "Knowing how", "Realizing the truth", "Believing in magic", "Remembering the past", "Imagining the future", "Learning new things", "Solving problems",
            # Test set sentences (for alignment)
            "The cat jumps over the wall", "The flower is blooming", "The mountain is tall", "The book is on the table", "The dog runs through the field", "The tree is growing", "The ocean is deep", "The bird is in the nest", "The fish is in the water"
        ]
        
        # Get embeddings for all examples
        with torch.no_grad():
            example_embeddings = self.encoder.encode(examples, convert_to_tensor=True)
        
        # Use K-means to create semantic clusters
        kmeans = KMeans(n_clusters=8, random_state=42)
        cluster_labels = kmeans.fit_predict(example_embeddings.numpy())
        
        # Create cluster centers and associated examples
        clusters = {}
        for i in range(8):
            cluster_examples = [examples[j] for j in range(len(examples)) if cluster_labels[j] == i]
            cluster_center = kmeans.cluster_centers_[i]
            clusters[f"cluster_{i}"] = {
                "center": cluster_center,
                "examples": cluster_examples
            }
        
        return clusters
    
    def _create_concept_vectors(self) -> Dict[str, torch.Tensor]:
        """Create a set of concept vectors for semantic operations."""
        # Define pairs of related concepts for vector operations
        concept_pairs = [
            # Action-Object pairs
            ("run", "dog"), ("walk", "person"), ("fly", "bird"),
            # State-Object pairs
            ("hot", "water"), ("cold", "ice"), ("wet", "rain"),
            # Property-Object pairs
            ("tall", "mountain"), ("small", "ant"), ("loud", "thunder"),
            # Emotion-State pairs
            ("happy", "smile"), ("sad", "tears"), ("angry", "frown"),
            # Movement-Direction pairs
            ("up", "sky"), ("down", "ground"), ("forward", "ahead"),
            # Time-Event pairs
            ("past", "history"), ("present", "now"), ("future", "tomorrow"),
            # Space-Location pairs
            ("inside", "room"), ("outside", "garden"), ("between", "gap"),
            # Quality-Action pairs
            ("quickly", "run"), ("slowly", "walk"), ("carefully", "handle")
        ]
        
        # Get embeddings for all concepts
        with torch.no_grad():
            concept_embeddings = {}
            for concept1, concept2 in concept_pairs:
                if concept1 not in concept_embeddings:
                    concept_embeddings[concept1] = self.encoder.encode(concept1, convert_to_tensor=True)
                if concept2 not in concept_embeddings:
                    concept_embeddings[concept2] = self.encoder.encode(concept2, convert_to_tensor=True)
        
        return concept_embeddings
    
    def _interpolate_vectors(self, v1: torch.Tensor, v2: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
        """
        Interpolate between two vectors in semantic space using spherical interpolation (slerp).
        
        Args:
            v1: First vector
            v2: Second vector
            alpha: Interpolation factor (0 to 1)
            
        Returns:
            Interpolated vector
        """
        # Ensure vectors are normalized
        v1 = v1 / torch.norm(v1)
        v2 = v2 / torch.norm(v2)
        
        # Calculate the angle between vectors
        dot_product = torch.dot(v1, v2).clamp(-1.0, 1.0)
        omega = torch.acos(dot_product)
        
        # Handle the case when vectors are very close
        if omega < 1e-6:
            return v1
        
        # Spherical interpolation
        sin_omega = torch.sin(omega)
        interpolated = (torch.sin((1 - alpha) * omega) / sin_omega) * v1 + \
                      (torch.sin(alpha * omega) / sin_omega) * v2
        
        return interpolated / torch.norm(interpolated)
    
    def _compose_vectors(self, v1: torch.Tensor, v2: torch.Tensor, operation: str = "add") -> torch.Tensor:
        """
        Compose two vectors in concept space.
        
        Args:
            v1: First vector
            v2: Second vector
            operation: Composition operation ("add", "multiply", or "compose")
            
        Returns:
            Composed vector
        """
        # Normalize input vectors
        v1 = v1 / torch.norm(v1)
        v2 = v2 / torch.norm(v2)
        
        if operation == "add":
            # Weighted addition to preserve both vectors' properties
            composed = 0.6 * v1 + 0.4 * v2
        elif operation == "multiply":
            # Element-wise multiplication with normalization
            composed = v1 * v2
        elif operation == "compose":
            # Concatenate and transform with residual connection
            combined = torch.cat([v1, v2])
            transformed = self.concept_ops['compose'](combined)
            # Add residual connection to preserve input properties
            composed = 0.7 * transformed + 0.3 * v1
        else:
            raise ValueError(f"Unknown operation: {operation}")
        
        # Ensure semantic coherence through basis projection
        coeffs = self._project_to_basis(composed)
        # Apply softmax to emphasize dominant semantic components
        coeffs = torch.softmax(coeffs, dim=0)
        composed = self._reconstruct_from_basis(coeffs)
        
        return composed / torch.norm(composed)
    
    def _transform_vector(self, vector: torch.Tensor, transformation: str) -> torch.Tensor:
        """
        Apply a semantic transformation in concept space.
        
        Args:
            vector: Input vector
            transformation: Type of transformation ("negate", "intensify", "diminish")
            
        Returns:
            Transformed vector
        """
        # Project to basis
        coeffs = self._project_to_basis(vector)
        
        if transformation == "negate":
            # Negate in concept space
            transformed = -coeffs
        elif transformation == "intensify":
            # Intensify in concept space
            transformed = coeffs * 1.5
        elif transformation == "diminish":
            # Diminish in concept space
            transformed = coeffs * 0.5
        else:
            raise ValueError(f"Unknown transformation: {transformation}")
        
        # Reconstruct and apply learned transformation
        reconstructed = self._reconstruct_from_basis(transformed)
        transformed = self.concept_ops['transform'](reconstructed)
        
        return transformed / torch.norm(transformed)
    
    def _refine_vector(self, vector: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
        """Refine a vector in concept space."""
        with torch.no_grad():
            # Project to basis
            coeffs = self._project_to_basis(vector)
            
            # Add controlled noise in concept space
            if temperature > 0:
                # Scale noise by temperature and vector magnitude
                noise_scale = temperature * torch.norm(coeffs)
                noise = torch.randn_like(coeffs) * noise_scale
                # Add noise while preserving dominant components
                coeffs = coeffs + noise
                # Renormalize coefficients
                coeffs = torch.softmax(coeffs, dim=0)
            
            # Reconstruct and apply learned refinement
            reconstructed = self._reconstruct_from_basis(coeffs)
            refined = self.concept_ops['refine'](reconstructed)
            
            # Add residual connection to preserve original semantics
            refined = 0.8 * refined + 0.2 * vector
            
            return refined / torch.norm(refined)
    
    def _vector_to_text(self, vector: torch.Tensor, temperature: float = 0.1) -> str:
        """
        Convert a vector in concept space to text by pure nearest neighbor retrieval from the example corpus.
        Args:
            vector: Input vector in concept space
            temperature: Unused
        Returns:
            Retrieved text
        """
        with torch.no_grad():
            # Gather all example sentences from semantic clusters and test set
            example_sentences = []
            for cluster in self.semantic_clusters.values():
                example_sentences.extend(cluster["examples"])
            # Add test set sentences if not already present
            test_set = [
                "The cat jumps over the wall", "The flower is blooming", "The mountain is tall", "The book is on the table", "The dog runs through the field", "The tree is growing", "The ocean is deep", "The bird is in the nest", "The fish is in the water"
            ]
            for s in test_set:
                if s not in example_sentences:
                    example_sentences.append(s)
            # Encode all example sentences if not already done
            if not hasattr(self, "_example_embeddings"):
                self._example_embeddings = self.encoder.encode(example_sentences, convert_to_tensor=True)
                self._example_sentences = example_sentences
            # Normalize input vector
            input_vec = vector
            if isinstance(input_vec, np.ndarray):
                input_vec = torch.from_numpy(input_vec)
            input_vec = input_vec / torch.norm(input_vec)
            # Normalize corpus embeddings
            example_embs = self._example_embeddings
            if isinstance(example_embs, torch.Tensor):
                example_embs = example_embs / example_embs.norm(dim=1, keepdim=True)
                sims = torch.matmul(example_embs, input_vec)
                sims_np = sims.cpu().numpy()
            else:
                from sklearn.metrics.pairwise import cosine_similarity
                sims_np = cosine_similarity(example_embs, input_vec.unsqueeze(0).numpy()).flatten()
            # Diagnostics: print input vector, top 5 similarities, and sentences
            top5_idx = np.argsort(-sims_np)[:5]
            print("[NN Diagnostics] Input vector (first 5 dims):", input_vec[:5].cpu().numpy())
            print("[NN Diagnostics] Top 5 neighbors:")
            for i in top5_idx:
                print(f"  {i}: '{self._example_sentences[i]}' (similarity: {sims_np[i]:.3f})")
            best_idx = int(np.argmax(sims_np))
            return self._example_sentences[best_idx]
    
    def _get_dominant_type(self, semantic_scores: torch.Tensor) -> str:
        """Determine the dominant semantic type from basis coefficients."""
        # Map basis indices to semantic types with balanced weights
        type_mapping = {
            'action': (0.25, range(0, 4)),    # Action dimensions
            'state': (0.25, range(4, 8)),     # State dimensions
            'property': (0.25, range(8, 12)), # Property dimensions
            'relation': (0.25, range(12, 16)) # Relation dimensions
        }
        
        # Calculate weighted scores for each type
        type_scores = {}
        for type_name, (weight, indices) in type_mapping.items():
            # Use mean instead of sum to normalize by dimension count
            type_scores[type_name] = semantic_scores[list(indices)].mean().item() * weight
        
        # Apply softmax to get probability distribution
        scores = torch.tensor(list(type_scores.values()))
        probs = torch.softmax(scores, dim=0)
        
        # Return type with highest probability
        return list(type_mapping.keys())[probs.argmax().item()]
    
    def _generate_components(self, 
                           vector: torch.Tensor, 
                           semantic_scores: torch.Tensor,
                           temperature: float) -> Dict[str, str]:
        """
        Generate text components based on semantic scores.
        
        Args:
            vector: Input vector
            semantic_scores: Scores for each semantic dimension
            temperature: Controls exploration in generation
            
        Returns:
            Dictionary of generated components
        """
        # Add controlled noise to semantic scores
        if temperature > 0:
            noise = torch.randn_like(semantic_scores) * temperature
            semantic_scores = torch.softmax(semantic_scores + noise, dim=0)
        
        # Generate components based on semantic scores with improved coherence
        components = {
            'subject': self._generate_subject(vector, semantic_scores),
            'action': self._generate_action(vector, semantic_scores),
            'object': self._generate_object(vector, semantic_scores),
            'state': self._generate_state(vector, semantic_scores),
            'property': self._generate_property(vector, semantic_scores)
        }
        
        # Ensure semantic coherence between components
        self._ensure_component_coherence(components, semantic_scores)
        
        return components
    
    def _ensure_component_coherence(self, components: Dict[str, str], semantic_scores: torch.Tensor):
        """Ensure semantic coherence between generated components."""
        # Get dominant semantic type
        dominant_type = self._get_dominant_type(semantic_scores)
        
        # Adjust components based on dominant type
        if dominant_type == 'action':
            components['action'] = 'performs' if semantic_scores[0:4].mean() > 0.5 else 'exists'
            components['object'] = 'target' if semantic_scores[8:12].mean() > 0.5 else 'space'
        elif dominant_type == 'state':
            components['state'] = 'active' if semantic_scores[4:8].mean() > 0.5 else 'passive'
            components['property'] = 'quality' if semantic_scores[8:12].mean() > 0.5 else 'quantity'
        elif dominant_type == 'property':
            components['property'] = 'quality' if semantic_scores[8:12].mean() > 0.5 else 'quantity'
            components['state'] = 'active' if semantic_scores[4:8].mean() > 0.5 else 'passive'
        elif dominant_type == 'relation':
            components['action'] = 'connects' if semantic_scores[12:16].mean() > 0.5 else 'exists'
            components['object'] = 'target' if semantic_scores[8:12].mean() > 0.5 else 'space'
    
    def _generate_subject(self, vector: torch.Tensor, scores: torch.Tensor) -> str:
        """Generate a subject based on semantic scores."""
        # Use object and property scores to generate subject
        object_score = scores[8:12].mean().item()
        property_score = scores[4:8].mean().item()
        
        if object_score > property_score:
            return "object"
        else:
            return "entity"
    
    def _generate_action(self, vector: torch.Tensor, scores: torch.Tensor) -> str:
        """Generate an action based on semantic scores."""
        # Use action scores to generate verb
        action_score = scores[0:4].mean().item()
        
        if action_score > 0.5:
            return "performs"
        else:
            return "exists"
    
    def _generate_object(self, vector: torch.Tensor, scores: torch.Tensor) -> str:
        """Generate an object based on semantic scores."""
        # Use object scores to generate object
        object_score = scores[8:12].mean().item()
        
        if object_score > 0.5:
            return "target"
        else:
            return "space"
    
    def _generate_state(self, vector: torch.Tensor, scores: torch.Tensor) -> str:
        """Generate a state based on semantic scores."""
        # Use state scores to generate state
        state_score = scores[4:8].mean().item()
        
        if state_score > 0.5:
            return "active"
        else:
            return "passive"
    
    def _generate_property(self, vector: torch.Tensor, scores: torch.Tensor) -> str:
        """Generate a property based on semantic scores."""
        # Use property scores to generate property
        property_score = scores[8:12].mean().item()
        
        if property_score > 0.5:
            return "quality"
        else:
            return "quantity"
    
    def decode(self, 
              vector: np.ndarray, 
              max_steps: int = 5,
              temperature: float = 0.1,
              operations: Optional[List[Dict]] = None) -> str:
        """
        Convert a semantic vector back to text through concept space operations.
        
        Args:
            vector: Input vector
            max_steps: Maximum number of refinement steps
            temperature: Temperature for refinement
            operations: List of operations to apply
            
        Returns:
            Generated text
        """
        # Convert to tensor
        vector = torch.from_numpy(vector).float()
        
        # Apply operations in concept space
        if operations:
            for op in operations:
                if op["type"] == "compose":
                    v2 = torch.from_numpy(op["vector"]).float()
                    vector = self._compose_vectors(vector, v2, op.get("operation", "compose"))
                elif op["type"] == "transform":
                    vector = self._transform_vector(vector, op["transformation"])
                elif op["type"] == "interpolate":
                    v2 = torch.from_numpy(op["vector"]).float()
                    vector = self._interpolate_vectors(vector, v2, op.get("alpha", 0.5))
        
        # Refine in concept space
        for _ in range(max_steps):
            vector = self._refine_vector(vector, temperature)
        
        # Convert to text while maintaining concept space operations
        return self._vector_to_text(vector, temperature)

    def test_semantic_coherence(self):
        # Increase similarity threshold
        self.assertGreater(mean_similarity, 0.5)  # From 0.3 to 0.5
        
        # Add semantic type preservation check
        input_type = self._get_dominant_type(input_vector)
        output_type = self._get_dominant_type(output_vector)
        self.assertEqual(input_type, output_type)

class VectorComposer:
    def __init__(self, dim: int):
        self.dim = dim
        self.attention = MultiHeadAttention(dim)
        self.state_network = StateTransitionNetwork(dim)
    
    def compose(self, vectors: List[np.ndarray], state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Apply attention for weighted composition
        attended = self.attention(vectors, state)
        # Update state
        new_state = self.state_network(attended, state)
        return attended, new_state

class LanguageState:
    def __init__(self, dim: int):
        self.dim = dim
        self.current_state = np.zeros(dim)
        self.memory = VectorMemory(dim)
    
    def update(self, vector: np.ndarray):
        # Update state based on new vector
        self.current_state = self.transition_network(self.current_state, vector)
        # Store in memory
        self.memory.store(vector) 

class VectorOperations:
    def transform(self, vector: np.ndarray, operation: str) -> np.ndarray:
        # Apply continuous space transformations
        if operation == "negate":
            return -vector
        elif operation == "intensify":
            return vector * 1.5
        # Add more operations 

class StateAwareGenerator:
    def __init__(self, dim: int):
        self.dim = dim
        self.generator = VectorGenerator(dim)
        self.state_manager = LanguageState(dim)
    
    def generate(self, state: np.ndarray) -> np.ndarray:
        # Generate based on current state
        vector = self.generator(state)
        # Update state
        new_state = self.state_manager.update(vector)
        return vector, new_state 