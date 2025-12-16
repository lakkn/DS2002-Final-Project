from typing import Any, Dict, List

from .embeddings import embed_texts
from .ingest import ingest_covid_data
from .llm import LocalLLM
from .retriever import Retriever
from .vector_store import build_and_save_index, load_index_and_docs

#rag file to connect everything together
class RAGPipeline:
    def __init__(self, retriever: Retriever, llm: LocalLLM):
        self.retriever = retriever
        self.llm = llm

    @classmethod
    def build_index(cls, data_dir: str) -> None:
        """
        building the embedding index after loading the directory
        """
        documents = ingest_covid_data(data_dir=data_dir)
        texts = [doc.text for doc in documents]
        embeddings = embed_texts(texts)
        build_and_save_index(documents, embeddings)

    @classmethod
    def from_artifacts(cls) -> "RAGPipeline":
        """
        retrieve the index and documents and init the llm
        """
        index, documents = load_index_and_docs()
        retriever = Retriever(index=index, documents=documents)
        llm = LocalLLM()
        return cls(retriever=retriever, llm=llm)


    def answer(self, question: str, k: int = 5) -> Dict[str, Any]:
        """
        final rag pipeline:
        retrieve context
        build prompt call LLM
        return answer
        """
        results = self.retriever.retrieve(question, k=max(k, 3))

        # textual context + source metadata building
        context_chunks: List[str] = []
        sources_by_file: Dict[str, Dict[str, Any]] = {}

        for i, (doc, score) in enumerate(results, start=1):
            src_id = doc.metadata.get("source_file", doc.id)
            context_chunks.append(
                f"[{i}] source={src_id}, doc_id={doc.id}, score={score:.3f}\n{doc.text}"
            )

            if src_id not in sources_by_file:
                sources_by_file[src_id] = {
                    "id": src_id,
                    "score": score,
                    "snippet": doc.text[:400],
                }

        context = "\n\n".join(context_chunks)

        system_prompt = (
            "You are an expert assistant that answers questions about COVID datasets. "
            "Use ONLY the information in the context below. "
            "If the answer is not in the context, say you do not know. "
            "Cite which chunks you used using [1], [2], etc. in your answer."
        )

        prompt = (
            f"{system_prompt}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            f"Answer:"
        )

        answer_text = self.llm.generate(prompt)
        sources = list(sources_by_file.values())

        return {
            "answer": answer_text,
            "sources": sources,
        }

# init with arguments to have more control over running the scripts
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--build-index",
        action="store_true",
        help="Ingest ./data CSVs and (re)build the FAISS index.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing COVID CSVs (default: data).",
    )
    args = parser.parse_args()

    if args.build_index:
        RAGPipeline.build_index(data_dir=args.data_dir)
        print("Finished building index.")
