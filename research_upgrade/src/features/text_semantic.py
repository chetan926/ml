"""
Semantic Feature Extractor (Phase 3.2, Text View 3)
Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection
"""

import os
from typing import Dict, Any, List, Optional
import numpy as np

# Enforce offline mode to prevent blocking on network calls
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class TextSemanticExtractor:
    """
    Extracts dense semantic embedding cosine similarity using all-MiniLM-L6-v2.
    
    Methodological Note:
    Dense semantic embeddings capture conceptual similarity and paraphrastic alignment.
    Semantic similarity alone does NOT constitute proof of academic plagiarism.
    It operates as one component of an uncertainty-aware multi-view screening ensemble.
    """

    MODEL_NAME = "all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384

    def __init__(self, model_name: str = MODEL_NAME, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model: Optional[Any] = None
        self._load_model()

    def _load_model(self) -> None:
        """Loads SentenceTransformer offline from local cache."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers package is required for TextSemanticExtractor."
            )
        try:
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                local_files_only=True
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load cached SentenceTransformer '{self.model_name}': {e}"
            )

    def extract(self, ref_text: str, sus_text: str) -> Dict[str, float]:
        """Extracts semantic similarity feature for a single text pair."""
        ref_str = str(ref_text).strip() if ref_text is not None else ""
        sus_str = str(sus_text).strip() if sus_text is not None else ""

        if not ref_str or not sus_str:
            return {"text_semantic_minilm_sim": 0.0}

        try:
            embs = self._model.encode(
                [ref_str, sus_str],
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            # Dot product of normalized unit vectors = cosine similarity in [-1, 1]
            raw_cosine = float(np.dot(embs[0], embs[1]))
            # Bound cleanly to [0, 1] via positive truncation / clamp
            bounded_sim = max(0.0, min(1.0, raw_cosine))
        except Exception:
            bounded_sim = 0.0

        return {
            "text_semantic_minilm_sim": round(bounded_sim, 6)
        }

    def extract_batch(self, ref_texts: List[str], sus_texts: List[str]) -> List[float]:
        """Batched encoding for high throughput during evaluation."""
        assert len(ref_texts) == len(sus_texts), "Batch size mismatch"
        if not ref_texts:
            return []

        clean_refs = [str(t).strip() if t is not None else "" for t in ref_texts]
        clean_suss = [str(t).strip() if t is not None else "" for t in sus_texts]

        # Filter empty indices
        valid_indices = [
            i for i, (r, s) in enumerate(zip(clean_refs, clean_suss)) if r and s
        ]

        similarities = [0.0] * len(ref_texts)
        if not valid_indices:
            return similarities

        sub_refs = [clean_refs[i] for i in valid_indices]
        sub_suss = [clean_suss[i] for i in valid_indices]

        embs_ref = self._model.encode(
            sub_refs,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        embs_sus = self._model.encode(
            sub_suss,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        dots = np.sum(embs_ref * embs_sus, axis=1)
        for idx, sim in zip(valid_indices, dots):
            similarities[idx] = round(float(max(0.0, min(1.0, sim))), 6)

        return similarities

    @property
    def feature_names(self) -> List[str]:
        return ["text_semantic_minilm_sim"]

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.EMBEDDING_DIM,
            "offline_mode": True,
            "device": self.device
        }

