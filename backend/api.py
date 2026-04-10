from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from chat_service import generate_reply
from schemas import ChatRequest
from question_processor import rewrite_question
from vector_store import ingest_data, retrieve_context


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event() -> None:
    ingest_data()


@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        processed_question = rewrite_question(request.message, request.history)
        context_docs = retrieve_context(processed_question)
        context = "\n".join(context_docs) if context_docs else ""

        reply = generate_reply(processed_question, request.history, context)
        return {"reply": reply}
    except Exception as exception:
        raise HTTPException(status_code=500, detail=str(exception))