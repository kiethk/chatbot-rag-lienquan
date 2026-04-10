import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
import uvicorn
from typing import Literal

# 1. Tải cấu hình và khởi tạo OpenAI (GitHub Models)
load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=os.environ.get("GITHUB_TOKEN"),
)

# 2. Khởi tạo FastAPI
app = FastAPI()

# 3. Cấu hình Middleware (CORS) - Quan trọng để Frontend kết nối được
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả các nguồn, hoặc ông đổi thành ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],  # Cho phép GET, POST, v.v.
    allow_headers=["*"],
)

# 4. Setup ChromaDB
db_client = chromadb.PersistentClient(path=os.path.join(BASE_DIR, "db_lienquan"))
default_ef = embedding_functions.DefaultEmbeddingFunction()
collection = db_client.get_or_create_collection(name="characters", embedding_function=default_ef)

# 5. Khai báo kiểu dữ liệu cho Request
class HistoryMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[HistoryMessage] = Field(default_factory=list)

# 6. Hàm nạp dữ liệu (Giữ nguyên logic cũ)
def ingest_data():
    if collection.count() > 0:
        return
    
    data_file = os.path.join(BASE_DIR, "data-character.json")
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for hero in data:
        content = (
            f"Tướng: {hero['name']}. Vai trò: {hero['role']}. Mô tả: {hero['description']} "
            f"Chiêu 1: {hero['skill1']}. Chiêu 2: {hero['skill2']}. Chiêu cuối: {hero['skill3']}. "
            f"Điểm mạnh: {hero['strengths']}. Điểm weakness: {hero['weaknesses']}. Khắc chế: {hero['counters']}."
        )
        collection.add(
            documents=[content],
            metadatas=[{"name": hero['name']}],
            ids=[f"hero_{hero['id']}"]
        )
    print("Dữ liệu đã sẵn sàng!")

# Nạp dữ liệu ngay khi khởi động server
ingest_data()

# 7. Endpoint chính: /api/chat
@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        # 1. Lấy ngữ cảnh từ RAG
        results = collection.query(query_texts=[request.message], n_results=3)
        documents = results.get("documents", [[]])
        context = "\n".join(documents[0]) if documents and documents[0] else ""

        # 2. Xây dựng danh sách tin nhắn bao gồm lịch sử
        messages = [
            {
                "role": "system",
                "content": "Bạn là trợ lý game Liên Quân. Trả lời ngắn gọn, súc tích dựa trên ngữ cảnh."
                + (f"\n\nNgữ cảnh:\n{context}" if context else ""),
            }
        ]

        # Thêm lịch sử (giới hạn 5 câu gần nhất để tiết kiệm token)
        for msg in request.history[-5:]:
            messages.append({"role": msg.role, "content": msg.content})

        # Thêm câu hỏi hiện tại
        messages.append({"role": "user", "content": request.message})

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.5,
        )
        return {"reply": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 8. Chạy server ở port 8000
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)