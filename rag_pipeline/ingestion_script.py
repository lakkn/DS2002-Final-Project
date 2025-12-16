
# import langchain document loaders
from langchain_community.document_loaders import PyPDFLoader, CSVLoader

# load PDF
pdf_loader = PyPDFLoader("C:\\Users\\micai\\OneDrive - University of Virginia\\Desktop\\DS 2002\\Coding For Final Project\\coping-with-covid19-conflict-afghanistan.pdf")
pdf_docs = pdf_loader.load()

# load CSV
csv_loader = CSVLoader("C:\\Users\\micai\\OneDrive - University of Virginia\\Desktop\\DS 2002\\Coding For Final Project\\Afghanistan_2020-01-01_to_2025-11-09.csv")
csv_docs = csv_loader.load()

# combine
all_docs = pdf_docs + csv_docs
print(f"Loaded {len(all_docs)} documents.")
print(all_docs[0].page_content[:520])  # Preview first doc
