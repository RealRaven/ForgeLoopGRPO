"""CPU/GPU embedding store optimized for Ryzen 9 7950X3D L3 cache."""

import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union, Optional


class CPUEmbeddingStore:
    """Embedding computation with bounded contiguous cache and batch processing.

    Optimized for high-cache CPU architectures (e.g., Ryzen 9 7950X3D).
    The bounded contiguous cache minimizes memory fragmentation and keeps hot
    embeddings inside L3 cache boundaries.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        max_cache_size: Optional[int] = None,
        thread_count: Optional[int] = None,
    ):
        # Optional thread-limit for asymmetric multi-CCD CPUs (e.g., 7950X3D).
        # NOTE: True CCD pinning requires OS-level affinity (taskset / sched_setaffinity);
        #       limiting thread count is a lightweight heuristic.
        if thread_count is not None:
            os.environ["OMP_NUM_THREADS"] = str(thread_count)
            os.environ["MKL_NUM_THREADS"] = str(thread_count)
            torch.set_num_threads(thread_count)

        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        self.model.eval()
        try:
            self.emb_dim = self.model.get_embedding_dimension()
        except AttributeError:
            self.emb_dim = self.model.get_sentence_embedding_dimension()

        # Bound cache to ~80 MB by default so model weights + cache + activations
        # stay inside a 96 MB L3 window (e.g., 7950X3D CCD0).
        if max_cache_size is None:
            bytes_per_entry = self.emb_dim * np.dtype(np.float32).itemsize
            max_cache_size = max(1024, int(80_000_000 // bytes_per_entry))
        self.max_cache_size = max_cache_size

        # Contiguous pre-allocated cache block for spatial locality
        self._cache_matrix = np.zeros((max_cache_size, self.emb_dim), dtype=np.float32)
        self._cache_map: dict = {}
        self._slot_to_key: List[Optional[str]] = [None] * max_cache_size
        self._next_slot = 0

        self._cache_hits = 0
        self._cache_misses = 0

    def encode(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        with torch.inference_mode():
            return self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

    def encode_cached(self, texts: List[str]) -> np.ndarray:
        n = len(texts)
        out_embeddings = np.empty((n, self.emb_dim), dtype=np.float32)

        uncached_texts = []
        uncached_indices = []

        # Step 1: Gather cached rows via index lookup
        for i, text in enumerate(texts):
            slot = self._cache_map.get(text)
            if slot is not None:
                out_embeddings[i] = self._cache_matrix[slot]
                self._cache_hits += 1
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
                self._cache_misses += 1

        # Step 2: Deduplicate uncached texts and encode only unique ones
        if uncached_texts:
            # Map each unique text to every output index that needs it
            text_to_indices: dict = {}
            for text, idx in zip(uncached_texts, uncached_indices):
                text_to_indices.setdefault(text, []).append(idx)

            unique_texts = list(text_to_indices.keys())
            unique_embeddings = self.encode(unique_texts)

            # Step 3: Broadcast each unique embedding to all requesting indices
            # and cache it exactly once via FIFO ring buffer
            for text, emb in zip(unique_texts, unique_embeddings):
                for idx in text_to_indices[text]:
                    out_embeddings[idx] = emb

                slot = self._next_slot
                old_key = self._slot_to_key[slot]
                if old_key is not None and old_key in self._cache_map:
                    del self._cache_map[old_key]

                self._cache_matrix[slot] = emb
                self._cache_map[text] = slot
                self._slot_to_key[slot] = text
                self._next_slot = (slot + 1) % self.max_cache_size

        return out_embeddings

    def gram_matrix(self, texts: List[str]) -> np.ndarray:
        embeddings = self.encode_cached(texts)
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        return np.dot(embeddings, embeddings.T)

    def max_pairwise_similarity(self, texts: List[str]) -> float:
        if len(texts) < 2:
            return 0.0
        gram = self.gram_matrix(texts)
        np.fill_diagonal(gram, -1.0)
        return float(np.max(gram))

    def cache_stats(self) -> dict:
        total = self._cache_hits + self._cache_misses
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": self._cache_hits / total if total > 0 else 0.0,
            "size": len(self._cache_map),
            "max_size": self.max_cache_size,
            "emb_dim": self.emb_dim,
        }