import os
import re
import random
import numpy as np
from typing import List, Tuple, Optional

class SimpleExampleRAG:
    """
    A lightweight RAG system to retrieve sales examples for style guidance.
    """
    def __init__(self, file_path: str, embedding_fn):
        self.file_path = file_path
        self.embedding_fn = embedding_fn
        self.examples: List[Tuple[str, str]] = [] # (User Query, Agent Response)
        self.vectors: Optional[np.ndarray] = None
        self.is_ready = False

    async def load_examples(self):
        if not os.path.exists(self.file_path):
            print(f"Error: Examples file not found at {self.file_path}")
            return

        try:
            with open(self.file_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            
            # Parse format: "1. User: ... Agent: ..."
            # Split by numbered lists or double newlines
            raw_blocks = re.split(r'\n\d+\.\s+User:', text)
            
            parsed = []
            for block in raw_blocks:
                if not block.strip(): continue
                # Look for "Agent:" split
                parts = block.split("Agent:", 1)
                if len(parts) == 2:
                    user_q = parts[0].strip(' "”\n')
                    agent_a = parts[1].strip(' "”\n')
                    # Clean up
                    user_q = re.sub(r'^\s*User:\s*', '', user_q, flags=re.IGNORECASE).strip(' "')
                    parsed.append((user_q, agent_a))
            
            self.examples = parsed
            print(f"Loaded {len(self.examples)} sales examples.")
            
            # Pre-compute embeddings for User queries
            vectors = []
            for q, _ in self.examples:
                vec = await self.embedding_fn(q)
                if vec:
                    vectors.append(vec)
                else:
                    vectors.append([0.0]*768) # Placeholder
            
            if vectors:
                self.vectors = np.array(vectors)
                self.is_ready = True
                print("Sales Examples Embeddings Initialized.")
                
        except Exception as e:
            print(f"Error loading examples: {e}")

    async def retrieve(self, query: str, k: int=1) -> str:
        if not self.is_ready or self.vectors is None:
            return ""
            
        try:
            query_vec = await self.embedding_fn(query)
            if not query_vec: return ""
            
            # from sklearn.metrics.pairwise import cosine_similarity
            # Manual Cosine Similarity with Numpy
            # scores = cosine_similarity([query_vec], self.vectors)[0]
            
            # Normalize query vector
            norm_q = np.linalg.norm(query_vec)
            if norm_q == 0: return ""
            query_vec = query_vec / norm_q
            
            # Vectors are already normalized? No, let's normalize them on load or here.
            # Assuming not normalized.
            norms_v = np.linalg.norm(self.vectors, axis=1)
            norms_v[norms_v == 0] = 1e-10 # Avoid init div by zero
            
            dot_products = np.dot(self.vectors, query_vec)
            scores = dot_products / norms_v
            
            # Get top K indices
            top_indices = scores.argsort()[-k:][::-1]
            
            result_str = ""
            for idx in top_indices:
                u, a = self.examples[idx]
                # Return only the agent's response to keep the LLM focused on style, 
                # not the meta-labels like "User:" and "Agent:"
                result_str += f"{a}\n"
                
            return result_str.strip()
            
        except Exception as e:
            print(f"Retrieval error: {e}")
            return ""
