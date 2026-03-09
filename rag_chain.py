from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

from dotenv import load_dotenv
load_dotenv()

import os 
from langchain_groq import ChatGroq

llm = ChatGroq(model = "llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))

#Retriever
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest

@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    """Inject context into state messages"""
    last_query = request.state['messages'][-1].text
    retrieved_docs = vector_store.similarity_search(last_query)

    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

    system_message = (
        "You are a helpful assistant that answers questions about Anas Benbaki. "
        "Use the following context from his CV in your response and be faithful to it:"
        f"\n\n{docs_content}"
        "Do not impersonate Anas or speak as him."
    )

    return system_message

agent = create_agent(llm, tools=[], middleware=[prompt_with_context])


