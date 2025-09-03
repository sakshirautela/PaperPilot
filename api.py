from fastapi import FastAPI, Query
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CHROMA_PATH = "chroma"
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="PaperPilot API",
    description="Semantic search over markdown documents using free embeddings.",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer_prompt: str
    sources: list

# Initialize embeddings and vector store once
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

@app.post("/query", response_model=QueryResponse)
def query_data(request: QueryRequest):
    results = db.similarity_search_with_relevance_scores(request.question, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        return QueryResponse(answer_prompt="Unable to find matching results.", sources=[])

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt = PROMPT_TEMPLATE.format(context=context_text, question=request.question)
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    return QueryResponse(answer_prompt=prompt, sources=sources)