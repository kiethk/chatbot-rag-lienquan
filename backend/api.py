from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from chat_service import generate_reply
from schemas import ChatRequest
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
        context_docs = retrieve_context(request.message)
        context = "\n".join(context_docs) if context_docs else ""

        reply = generate_reply(request.message, request.history, context)
        return {"reply": reply}
    except Exception as exception:
        raise HTTPException(status_code=500, detail=str(exception))