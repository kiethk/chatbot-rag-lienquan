from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# from config import GENERATION_TEMPERATURE, get_api_key, MODEL_NAME
from config import GENERATION_TEMPERATURE, LLM_API_KEY, LLM_BASE_URL, MODEL_NAME


_question_rewriter = ChatOpenAI(
    model=MODEL_NAME,
    temperature=GENERATION_TEMPERATURE,
    # base_url="https://models.inference.ai.azure.com",
    # api_key=get_api_key(),
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
)

_rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Bạn là bộ xử lý câu hỏi cho hệ thống RAG Liên Quân. "
            "Nhiệm vụ của bạn là chuyển câu hỏi hiện tại thành một câu hỏi độc lập, rõ nghĩa, "
            "giữ nguyên tên tướng, kỹ năng và thuật ngữ game nếu có. "
            "Nếu câu hỏi đã độc lập rồi thì chỉ chuẩn hóa nhẹ cho rõ nghĩa, không được thêm dữ kiện mới. "
            "Chỉ trả về đúng một câu hỏi đã rewrite, không giải thích.",
        ),
        (
            "human",
            "Lịch sử hội thoại gần nhất:\n{history}\n\nCâu hỏi hiện tại:\n{question}",
        ),
    ]
)

_rewrite_chain = _rewrite_prompt | _question_rewriter


def _format_history(history: list, limit: int = 4) -> str:
    if not history:
        return "(trống)"

    lines: list[str] = []
    for item in history[-limit:]:
        if getattr(item, "role", None) not in {"user", "assistant"}:
            continue
        lines.append(f"{item.role}: {item.content}")
    return "\n".join(lines) if lines else "(trống)"


def rewrite_question(question: str, history: list) -> str:
    formatted_history = _format_history(history)
    result = _rewrite_chain.invoke({"history": formatted_history, "question": question})
    rewritten = getattr(result, "content", str(result)).strip()
    return rewritten or question.strip()