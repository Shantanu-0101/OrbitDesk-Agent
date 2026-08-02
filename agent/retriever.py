"""
agent/retriever.py
------------------
Converts all document chunks into vectors using a local Hugging Face
embedding model, stores them in a FAISS index, and provides a search
function that finds the most relevant passages for any query.

HOW IT WORKS:
  1. Load all 50 document chunks from loader.py
  2. Use sentence-transformers (all-MiniLM-L6-v2) to embed each chunk
     into a 384-dimensional vector
  3. Store all vectors in a FAISS index (in-memory, no external DB)
  4. At query time: embed the question → find top-K nearest vectors
  5. Return the corresponding Document objects with similarity scores
"""

import os
import time
import numpy as np
import faiss
from typing import List, Tuple
from sentence_transformers import SentenceTransformer

from agent.loader import Document, load_all_documents


#  Configuration 

# Model name — must be a Hugging Face sentence-transformers model
# all-MiniLM-L6-v2 is fast, small (~90MB), and works great on CPU
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# How many top results to return by default
DEFAULT_TOP_K = 5


#  Retriever Class 

class Retriever:
    """
    Manages the embedding model and FAISS index.

    Usage:
        retriever = Retriever()
        results = retriever.search("How do I fix a missed export?", top_k=5)
    """

    def __init__(self):
        self.model: SentenceTransformer = None
        self.index: faiss.IndexFlatIP = None   # Inner Product = cosine similarity
        self.documents: List[Document] = []
        self.is_ready = False

    def build(self):
        """
        Full setup: load model → load documents → embed → build FAISS index.
        Call this once at startup. After this, search() works offline.
        """
        print("\n" + "="*55)
        print("  RETRIEVER SETUP")
        print("="*55)

        # ── Step 1: Load the embedding model 
        print(f"\n[1/3] Loading embedding model: {EMBEDDING_MODEL_NAME}")
        print("      (Downloads ~90MB on first run, cached after that)")
        t0 = time.time()

        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        load_time = time.time() - t0
        print(f"      Model loaded in {load_time:.1f}s")

        # ── Step 2: Load all documents 
        print("\n[2/3] Loading documents...")
        self.documents = load_all_documents()
        print(f"      {len(self.documents)} chunks ready for embedding")

        # ── Step 3: Embed all chunks + build FAISS index 
        print("\n[3/3] Embedding all chunks and building FAISS index...")
        t1 = time.time()

        # Get the raw text from every document chunk
        texts = [doc.text for doc in self.documents]

        # Embed all texts at once (batch processing = faster)
        # Shape: (num_docs, 384)
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # Required for cosine similarity via dot product
        )

        # Build FAISS index
        # IndexFlatIP = exact search using Inner Product (= cosine sim when normalized)
        dimension = embeddings.shape[1]           # 384 for MiniLM
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype(np.float32))

        embed_time = time.time() - t1
        print(f"\n      Indexed {self.index.ntotal} vectors in {embed_time:.1f}s")
        print(f"      Embedding dimension: {dimension}")

        self.is_ready = True
        print("\n  Retriever is ready!\n" + "="*55 + "\n")

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        exclude_superseded: bool = False
    ) -> List[Tuple[Document, float]]:
        """
        Search for the most relevant document chunks for a given query.

        Args:
            query            : The user's question (raw natural language)
            top_k            : How many results to return (default 5)
            exclude_superseded: If True, filters out CASE-0914 and other
                               superseded cases from results

        Returns:
            List of (Document, score) tuples, sorted by relevance (highest first)
            Score is cosine similarity: 1.0 = identical, 0.0 = unrelated
        """
        if not self.is_ready:
            raise RuntimeError("Retriever not built yet. Call retriever.build() first.")

        # Embed the query (normalize = True, same as documents)
        query_vector = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)

        # Search FAISS — returns (distances, indices) arrays
        # distances = cosine similarity scores (higher = more relevant)
        scores, indices = self.index.search(query_vector, top_k * 2)  # fetch extra for filtering

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:   # FAISS returns -1 when fewer results than top_k
                continue

            doc = self.documents[idx]

            # Optionally skip superseded cases
            if exclude_superseded and doc.metadata.get("is_superseded", False):
                continue

            results.append((doc, float(score)))

            if len(results) >= top_k:
                break

        return results

    def format_results(self, results: List[Tuple[Document, float]]) -> str:
        """
        Human-readable formatting of search results (for debugging/logs).
        """
        lines = []
        for i, (doc, score) in enumerate(results, 1):
            superseded = " [SUPERSEDED]" if doc.metadata.get("is_superseded") else ""
            lines.append(
                f"  [{i}] {doc.source_id}{superseded} (score: {score:.3f})\n"
                f"      {doc.text[:120].strip()}..."
            )
        return "\n".join(lines)


#  Singleton pattern
# We create ONE shared Retriever instance that gets reused across all nodes.
# Building it once at startup avoids re-loading the model on every question.

_retriever_instance: Retriever = None

def get_retriever() -> Retriever:
    """
    Returns the shared Retriever instance, building it if needed.
    This is called once at graph startup.
    """
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = Retriever()
        _retriever_instance.build()
    return _retriever_instance


#  Quick test 
if __name__ == "__main__":
    r = get_retriever()

    test_query = "What happens when I change the workspace timezone?"
    print(f"Query: {test_query}\n")

    results = r.search(test_query, top_k=3)
    print(r.format_results(results))