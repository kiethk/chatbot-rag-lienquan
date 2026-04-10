import os
import re
import unicodedata

import chromadb
from chromadb.utils import embedding_functions

from config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DATA_TEXT_FILE,
    RETRIEVAL_TOP_K,
)


db_client = chromadb.PersistentClient(path=CHROMA_DIR)
default_ef = embedding_functions.DefaultEmbeddingFunction()
collection = db_client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=default_ef,
)


def _normalize_text(text: str) -> str:
    lowered = text.lower()
    no_diacritics = "".join(
        ch for ch in unicodedata.normalize("NFD", lowered) if unicodedata.category(ch) != "Mn"
    )
    return re.sub(r"[^a-z0-9]+", "", no_diacritics)


def _slugify(text: str) -> str:
    normalized = _normalize_text(text)
    return normalized if normalized else "unknown"


def _load_raw_text() -> str:
    if not os.path.exists(DATA_TEXT_FILE):
        raise FileNotFoundError(f"Data text file not found: {DATA_TEXT_FILE}")
    with open(DATA_TEXT_FILE, "r", encoding="utf-8") as file_handle:
        return file_handle.read().strip()


def _split_hero_sections(raw_text: str) -> list[tuple[str, str]]:
    pattern = re.compile(r"^##\s*\d+\.\s*(.+)$", flags=re.MULTILINE)
    matches = list(pattern.finditer(raw_text))
    sections: list[tuple[str, str]] = []

    if not matches:
        return sections

    for index, match in enumerate(matches):
        hero_name = match.group(1).strip()
        body_start = match.end()
        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(raw_text)
        body = raw_text[body_start:body_end].strip()
        if body:
            sections.append((hero_name, body))
    return sections


def _chunk_text(text: str, max_chars: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    cleaned = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not cleaned:
        return []

    chunks: list[str] = []
    start = 0
    text_length = len(cleaned)

    while start < text_length:
        end = min(start + max_chars, text_length)

        # Cố gắng cắt tại ranh giới dòng để chunk tự nhiên hơn.
        if end < text_length:
            fallback = cleaned.rfind("\n", start + int(max_chars * 0.6), end)
            if fallback != -1 and fallback > start:
                end = fallback

        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break
        start = max(end - overlap, start + 1)

    return chunks


RAW_TEXT = _load_raw_text()
HERO_SECTIONS = _split_hero_sections(RAW_TEXT)
HERO_NAMES = [name for name, _ in HERO_SECTIONS]


def _extract_mentioned_heroes(question: str) -> list[str]:
    normalized_question = _normalize_text(question)
    return [name for name in HERO_NAMES if _normalize_text(name) in normalized_question]


def ingest_data() -> None:
    if collection.count() > 0:
        return

    documents: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []

    for hero_name, hero_text in HERO_SECTIONS:
        chunks = _chunk_text(hero_text)
        for chunk_index, chunk in enumerate(chunks):
            documents.append(f"Tướng: {hero_name}\n{chunk}")
            metadatas.append(
                {
                    "name": hero_name,
                    "source": "data-lienquan.txt",
                    "chunk_index": chunk_index,
                }
            )
            ids.append(f"hero_{_slugify(hero_name)}_chunk_{chunk_index}")

    if documents:
        collection.add(documents=documents, metadatas=metadatas, ids=ids)


def retrieve_context(question: str, top_k: int = RETRIEVAL_TOP_K) -> list[str]:
    docs: list[str] = []
    seen: set[str] = set()

    # Nếu user nhắc đích danh tướng thì ưu tiên lấy đúng tướng đó trước.
    for hero_name in _extract_mentioned_heroes(question):
        hero_result = collection.query(query_texts=[question], n_results=2, where={"name": hero_name})
        hero_docs = hero_result.get("documents", [[]])
        if hero_docs and hero_docs[0]:
            for doc in hero_docs[0]:
                if doc not in seen:
                    docs.append(doc)
                    seen.add(doc)
                if len(docs) >= top_k:
                    return docs[:top_k]

    semantic_result = collection.query(query_texts=[question], n_results=max(top_k * 2, 6))
    semantic_docs = semantic_result.get("documents", [[]])
    if semantic_docs and semantic_docs[0]:
        for doc in semantic_docs[0]:
            if doc not in seen:
                docs.append(doc)
                seen.add(doc)
            if len(docs) >= top_k:
                break

    return docs[:top_k]
