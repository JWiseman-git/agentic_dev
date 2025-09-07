"""
RAG job suitability pipeline V1.

The contxt provided:
1. Most up to date CV.
2. The role to benchmark against.

TODO: Build RAW LinkedIn scraper
"""

from langchain_community.document_loaders import PyPDFLoader

file_path = "../example_data/nke-10k-2023.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()

print(len(docs))