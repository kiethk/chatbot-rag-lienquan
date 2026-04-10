import os

from dotenv import load_dotenv


load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data-character.json")
DATA_TEXT_FILE = os.path.join(BASE_DIR, "data-lienquan.txt")
CHROMA_DIR = os.path.join(BASE_DIR, "db_lienquan")
COLLECTION_NAME = "characters_text_chunks"
MODEL_NAME = "gpt-4o-mini"
SYSTEM_PROMPT = (
	"Bạn là trợ lý game Liên Quân, chỉ được trả lời dựa trên ngữ cảnh được cung cấp. "
	"Không bịa thêm dữ kiện ngoài ngữ cảnh. "
	"Nếu ngữ cảnh không đủ để kết luận, hãy nói rõ: 'Mình chưa đủ dữ liệu trong kho RAG hiện tại để trả lời chính xác.' "
	"Ưu tiên trả lời ngắn gọn, đúng trọng tâm và không nêu các tướng không xuất hiện trong ngữ cảnh."
)
HISTORY_LIMIT = 5
RETRIEVAL_TOP_K = 3
GENERATION_TEMPERATURE = 0.1
CHUNK_SIZE = 500
CHUNK_OVERLAP = 120
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANKER_CANDIDATE_K = 8
RERANKER_TOP_K = 3


def get_api_key() -> str | None:
    return os.environ.get("GITHUB_TOKEN") or os.environ.get("OPENAI_API_KEY")