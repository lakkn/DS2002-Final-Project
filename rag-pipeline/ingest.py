from pathlib import Path
from typing import List

import pandas as pd

from .types import Document


def _row_to_text(row: pd.Series) -> str:
    """
    combining columns of a csv row to make it a string
    """
    parts = [f"{col}: {row[col]}" for col in row.index]
    return " | ".join(parts)


def ingest_covid_data(data_dir: str = "data/csv_data", rows_per_chunk: int = 20) -> List[Document]:
    """
    Ingest all CSVs in data_dir into chunked text Documents.
    """
    data_path = Path(data_dir)
    csv_files = sorted(data_path.glob("*.csv"))

    documents: List[Document] = []

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        if df.empty:
            continue

        for start in range(0, len(df), rows_per_chunk):
            end = min(start + rows_per_chunk, len(df))
            chunk_df = df.iloc[start:end]

            lines = [_row_to_text(row) for _, row in chunk_df.iterrows()]
            text = "\n".join(lines)

            doc_id = f"{csv_path.name}#{start}-{end - 1}"
            metadata = {
                "source_file": csv_path.name,
                "start_row": start,
                "end_row": end - 1,
                "num_rows": end - start,
            }

            documents.append(Document(id=doc_id, text=text, metadata=metadata))

    print(f"Ingested {len(documents)} documents from {len(csv_files)} CSV files.")
    return documents