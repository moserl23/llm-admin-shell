"""
Minimal RAG app (single file) with line-by-line explanations.
- Uses OpenAI for both the chat model and embeddings (only one API key required).
- Orchestrates retrieval + generation with LangGraph.
- Loads a single web page and answers a question about it.
"""

# ---------- 0) Imports & configuration ----------
import os  # standard lib: to set environment variables for API keys

# pull your OpenAI key from a local config; you said: `from config import API-KEY`
# Python identifiers cannot contain '-', so we assume it's `API_KEY` in your config.py
from config import API_KEY  # config.py should define API_KEY = "sk-..."

# LangChain core types and utilities
from typing import List  # for type hints
from typing_extensions import TypedDict  # for typed state in LangGraph

# Loader for web pages and BeautifulSoup for HTML parsing
import bs4  # HTML parsing engine used by WebBaseLoader filters
from langchain_community.document_loaders import WebBaseLoader  # loads URLs into Documents

# Document type (LangChain's simple text container with metadata)
from langchain_core.documents import Document

# Split long docs into chunks for embedding & retrieval
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vector store (in-memory) and OpenAI embeddings + chat model
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Prompt hub: tiny, battle-tested RAG template
from langchain import hub

# LangGraph for wiring steps into a tiny app graph
from langgraph.graph import START, StateGraph


# ---------- 1) Set API key & initialize models ----------
# Put your OpenAI key into the environment where langchain_openai looks for it.
os.environ["OPENAI_API_KEY"] = API_KEY  # only needed once per process

# Create a chat model. Use a lightweight, inexpensive OpenAI model.
# You can swap "gpt-4o-mini" with any other available OpenAI chat model.
llm = ChatOpenAI(model="gpt-4o-mini")  # LLM that writes the final answer

# Create an embeddings model for turning text → vectors (for vector search).
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Create an in-memory vector store that will hold our (embedding, text) pairs.
# InMemoryVectorStore is perfect for demos; swap to Pinecone/Qdrant/etc. in production.
vector_store = InMemoryVectorStore(embeddings)


# ---------- 2) Indexing (one-time: load → split → embed/store) ----------
# We’ll use a well-known public post about LLM agents (stable URL).
URL = "https://lilianweng.github.io/posts/2023-06-23-agent/"

# Configure BeautifulSoup to keep only relevant HTML sections (less noise → better chunks).
bs4_strainer = bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))

# Load the web page and return a list of `Document` objects (usually length 1 here).
loader = WebBaseLoader(web_paths=(URL,), bs_kwargs={"parse_only": bs4_strainer})
docs: List[Document] = loader.load()  # fetch & parse HTML → clean text in Document(s)

# Split the doc(s) into overlapping chunks so retrieval can find the right snippets.
#  - chunk_size ~1000 chars is a common default
#  - chunk_overlap keeps context continuity between chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits: List[Document] = text_splitter.split_documents(docs)

# Add all chunks to the vector store:
#  - this computes embeddings under the hood
#  - stores (embedding, text, metadata) for similarity search
_ = vector_store.add_documents(splits)


# ---------- 3) Prompt template for RAG ----------
# Pull a concise, safety-aware RAG prompt from LangChain’s prompt hub.
# It expects two variables:
#   - question: the user’s question
#   - context: the retrieved text snippets
prompt = hub.pull("rlm/rag-prompt")


# ---------- 4) Define app state & steps (LangGraph) ----------
# App state shape: what enters, flows between steps, and exits the graph.
class State(TypedDict):
    question: str            # user input
    context: List[Document]  # retrieved chunks
    answer: str              # final model answer

# Step 1: retrieve() finds the most relevant chunks for the question.
def retrieve(s: State) -> dict:
    # vector_store.similarity_search returns a small list of best-matching Documents
    retrieved_docs = vector_store.similarity_search(s["question"], k=4)
    # Return a partial state update: we’re filling the "context" field
    return {"context": retrieved_docs}

# Step 2: generate() feeds question + retrieved text into the LLM to produce an answer.
def generate(s: State) -> dict:
    # Stitch retrieved chunks into one context string (simple and effective)
    ctx = "\n\n".join(d.page_content for d in s["context"])
    # Format the RAG prompt with our variables → returns a ChatPromptValue/messages
    messages = prompt.invoke({"question": s["question"], "context": ctx})
    # Call the LLM with the prepared messages and get its reply
    resp = llm.invoke(messages)
    # Return another partial state update: we’re filling the "answer" field
    return {"answer": resp.content}

# Wire the two steps into a tiny graph:
#  - When invoked, it will run retrieve → generate in sequence.
graph_builder = StateGraph(State).add_sequence([retrieve, generate])

# Define the starting edge: execution starts at START and first calls "retrieve"
graph_builder.add_edge(START, "retrieve")

# Compile the graph into a runnable object that supports invoke/stream/async/batch
graph = graph_builder.compile()



# ---------- 5) Run it ----------
if __name__ == "__main__":
    # Ask something the page definitely discusses:
    user_question = "What is Task Decomposition?"
    # Execute the graph synchronously with our input
    result_state: State = graph.invoke({"question": user_question})

    # Print the final answer
    print("\n=== Answer ===")
    print(result_state["answer"])

    # (Optional) Show where the answer came from (the retrieved sources)
    print("\n=== Sources (top retrieved chunks) ===")
    for i, doc in enumerate(result_state["context"], start=1):
        src = doc.metadata.get("source", "unknown")
        print(f"[{i}] {src}  (chars: {len(doc.page_content)})")
