#Load
from langchain_community.document_loaders import PyPDFLoader

file_path = "~/Downloads/cv_fin.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()

#Split into text chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150, add_start_index=True)
all_splits = text_splitter.split_documents(docs)

#Embedding the text chunks(into vectors) + Store them
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

embedding_dim = len(embeddings.embed_query("hello world"))
index = faiss.IndexFlatL2(embedding_dim)

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

ids = vector_store.add_documents(documents=all_splits)
vector_store.save_local("faiss_index")

#Retriever
# from typing import List

# from langchain_core.documents import Document
# from langchain_core.runnables import chain

# @chain
# def retriever(query: str) -> List[Document]:
#     return vector_store.similarity_search(query, k=1)

# # retriever = vector_store.as_retriever(
# #     search_type="similarity",
# #     search_kwargs={"k" : 1},
# # )

