from dataclasses import dataclass
from typing import Any, Dict

# this is the document type that will be passed in as data for the RAG
@dataclass
class Document:
    id: str
    text: str
    metadata: Dict[str, Any]