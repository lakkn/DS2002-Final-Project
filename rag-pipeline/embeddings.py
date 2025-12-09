from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

_EMBEDDING_MODEL = None
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# load the embedding model if it doesnt exist
def get_embedding_model() -> SentenceTransformer:
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        _EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _EMBEDDING_MODEL

# returns embedded texts as np array
def embed_texts(texts: List[str]) -> np.ndarray:
    model = get_embedding_model()
    return np.asarray(model.encode(texts, show_progress_bar=True), dtype="float32")