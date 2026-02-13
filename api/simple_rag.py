import os
import re
import random
from typing import List, Tuple, Optional

class SimpleExampleRAG:
    """
    A lightweight RAG system to retrieve sales examples for style guidance.
    """
    def __init__(self, file_path: str, embedding_fn):
        self.file_path = file_path
        self.embedding_fn = embedding_fn
        self.examples: List[Tuple[str, str]] = [] # (User Query, Agent Response)
        self.vectors: Optional[List[List[float]]] = None
        self.is_ready = False

    async def load_examples(self):
        if not os.path.exists(self.file_path):
            print(f"Error: Examples file not found at {self.file_path}")
            return

        try:
            with open(self.file_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            
            raw_blocks = re.split(r'\n\d+\.\s+User:', text)
            
            parsed = []
            for block in raw_blocks:
                if not block.strip(): continue
                parts = block.split("Agent:", 1)
                if len(parts) == 2:
                    user_q = parts[0].strip(' "”\n')
                    agent_a = parts[1].strip(' "”\n')
                    user_q = re.sub(r'^\s*User:\s*', '', user_q, flags=re.IGNORECASE).strip(' "')
                    parsed.append((user_q, agent_a))
            
            self.examples = parsed
            print(f"Loaded {len(self.examples)} sales examples.")
            
            vectors = []
            for q, _ in self.examples:
                vec = await self.embedding_fn(q)
                if vec:
                    vectors.append(vec)
                else:
                    vectors.append([0.0]*768) # Placeholder
            
            if vectors:
                self.vectors = vectors
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
            
            def cosine_sim(v1, v2):
                dot = sum(a*b for a, b in zip(v1, v2))
                norm1 = sum(a*a for a in v1)**0.5
                norm2 = sum(a*a for a in v2)**0.5
                if norm1 == 0 or norm2 == 0: return 0
                return dot / (norm1 * norm2)

            scores = [cosine_sim(query_vec, v) for v in self.vectors]
            
            # Get top K indices
            indexed_scores = list(enumerate(scores))
            indexed_scores.sort(key=lambda x: x[1], reverse=True)
            top_indices = [idx for idx, score in indexed_scores[:k]]
            
            result_str = ""
            for idx in top_indices:
                u, a = self.examples[idx]
                result_str += f"{a}\n"
                
            return result_str.strip()
            
        except Exception as e:
            print(f"Retrieval error: {e}")
            return ""
