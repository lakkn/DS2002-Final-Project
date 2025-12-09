from pathlib import Path
import pickle
from typing import List, Tuple

import faiss
import numpy as np

from .types import Document

ARTIFACTS_DIR = Path("artifacts")
INDEX_PATH = ARTIFACTS_DIR / "covid_faiss.index"
DOCS_PATH = ARTIFACTS_DIR / "covid_docs.pkl"


def _ensure_artifacts_dir() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def build_and_save_index(documents: List[Document], embeddings: np.ndarray) -> None:
    """
    Build FAISS index with inner product similarity and save index to disk.
    """
    _ensure_artifacts_dir()

    # Normalize embeddings
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))

    with DOCS_PATH.open("wb") as f:
        pickle.dump(documents, f)

    print(f"FAISS index saved to {INDEX_PATH}")
    print(f"{len(documents)} documents saved at {DOCS_PATH}")


def load_index_and_docs() -> Tuple[faiss.Index, List[Document]]:
    """
    Load FAISS from disk.
    """
    _ensure_artifacts_dir()

    index = faiss.read_index(str(INDEX_PATH))
    with DOCS_PATH.open("rb") as f:
        documents: List[Document] = pickle.load(f)

    print(f"{index.ntotal} vectors loaded.")
    print(f"{len(documents)} documents loaded.")
    return index, documents