from openai import OpenAI

from config import (
    GENERATION_TEMPERATURE,
    HISTORY_LIMIT,
    LLM_API_KEY,
    LLM_BASE_URL,
    MODEL_NAME,
    SYSTEM_PROMPT,
)

client = OpenAI(
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
)


def build_messages(message: str, history: list, context: str) -> list[dict[str, str]]:
    system_message = SYSTEM_PROMPT
    if context:
        system_message += f"\n\nNgữ cảnh:\n{context}"

    messages = [{"role": "system", "content": system_message}]

    for msg in history[-HISTORY_LIMIT:]:
        if msg.role in {"user", "assistant"}:
            messages.append({"role": msg.role, "content": msg.content})

    messages.append({"role": "user", "content": message})
    return messages


def generate_reply(message: str, history: list, context: str) -> str:
    messages = build_messages(message, history, context)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=GENERATION_TEMPERATURE,
    )
    return response.choices[0].message.content