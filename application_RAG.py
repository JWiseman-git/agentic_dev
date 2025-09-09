"""
RAG job suitability pipeline V1.

The contxt provided:
1. Most up to date CV.
2. The role to benchmark against.

TODO: Build RAW LinkedIn scraper
"""
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

from typing import List
from langchain_core.documents import Document

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv
from pathlib import Path

from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

from langchain_core.runnables import chain

env_path = Path('.') / '.env'
load_dotenv(env_path)

# Load PDF
file_path = "./sample_data/Jordan_Wiseman_CV_AI_Engineer.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

sample_vector = embeddings.embed_query("test")
embedding_dim = len(sample_vector)
index = faiss.IndexFlatL2(embedding_dim)

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

ids = vector_store.add_documents(documents=all_splits)

results = vector_store.similarity_search_with_score(
    "Where does Jordan live?"
)
doc, score = results[0]
print(score)

# Retriever Factory Method
@chain
def retriever(query: str) -> List[Document]:
    return vector_store.similarity_search(query, k=1)

retriever.batch(
    [
        "How many distribution centers does Nike have in the US?",
        "When was Nike incorporated?",
    ],
)