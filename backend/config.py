import os

from dotenv import load_dotenv


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

DATA_FILE = os.path.join(BASE_DIR, "data-character.json")
DATA_TEXT_FILE = os.path.join(BASE_DIR, "data-lienquan.txt")
CHROMA_DIR = os.path.join(BASE_DIR, "db_lienquan")
COLLECTION_NAME = "characters_text_chunks"
MODEL_NAME = os.environ.get("MODEL_NAME", "gemma4:e4b")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "ollama")

SYSTEM_PROMPT = (
	"Bạn là trợ lý game Liên Quân, chỉ được trả lời dựa trên ngữ cảnh được cung cấp. "
	"Không bịa thêm dữ kiện ngoài ngữ cảnh. "
	"Nếu ngữ cảnh không đủ để kết luận, hãy nói rõ: 'Mình chưa đủ dữ liệu trong kho RAG hiện tại để trả lời chính xác.' "
	"Ưu tiên trả lời ngắn gọn, đúng trọng tâm và không nêu các tướng không xuất hiện trong ngữ cảnh."
)
HISTORY_LIMIT = int(os.environ.get("HISTORY_LIMIT", "5"))
RETRIEVAL_TOP_K = int(os.environ.get("RETRIEVAL_TOP_K", "3"))
GENERATION_TEMPERATURE = float(os.environ.get("GENERATION_TEMPERATURE", "0.1"))
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "120"))
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RERANKER_CANDIDATE_K = int(os.environ.get("RERANKER_CANDIDATE_K", "8"))
RERANKER_TOP_K = int(os.environ.get("RERANKER_TOP_K", "3"))