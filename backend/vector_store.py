import os
import re
import unicodedata

import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder

from config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DATA_TEXT_FILE,
    RERANKER_CANDIDATE_K,
    RERANKER_MODEL,
    RERANKER_TOP_K,
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


def _recursive_split(text: str, separators: list[str], max_chars: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    if not separators:
        return [text[i:i + max_chars].strip() for i in range(0, len(text), max_chars)]

    separator = separators[0]
    if separator == "":
        return [text[i:i + max_chars].strip() for i in range(0, len(text), max_chars)]

    parts = [part.strip() for part in text.split(separator)]
    parts = [part for part in parts if part]
    if len(parts) <= 1:
        return _recursive_split(text, separators[1:], max_chars)

    chunks: list[str] = []
    buffer = ""

    for part in parts:
        candidate = part if not buffer else f"{buffer}{separator}{part}"
        if len(candidate) <= max_chars:
            buffer = candidate
            continue

        if buffer:
            chunks.extend(_recursive_split(buffer, separators[1:], max_chars))
            buffer = part
        else:
            chunks.extend(_recursive_split(part, separators[1:], max_chars))

    if buffer:
        chunks.extend(_recursive_split(buffer, separators[1:], max_chars))

    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _apply_overlap(chunks: list[str], overlap: int, max_chars: int) -> list[str]:
    if not chunks or overlap <= 0:
        return chunks

    overlapped: list[str] = [chunks[0]]
    for index in range(1, len(chunks)):
        prefix = chunks[index - 1][-overlap:]
        merged = f"{prefix}\n{chunks[index]}".strip()
        overlapped.append(merged if len(merged) <= max_chars else chunks[index])
    return overlapped


def _chunk_text(text: str, max_chars: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    cleaned = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not cleaned:
        return []

    separators = ["\n\n", "\n", ". ", "; ", ", ", " ", ""]
    chunks = _recursive_split(cleaned, separators, max_chars)
    return _apply_overlap(chunks, overlap, max_chars)


RAW_TEXT = _load_raw_text()
HERO_SECTIONS = _split_hero_sections(RAW_TEXT)
HERO_NAMES = [name for name, _ in HERO_SECTIONS]

_reranker: CrossEncoder | None = None


def _extract_mentioned_heroes(question: str) -> list[str]:
    normalized_question = _normalize_text(question)
    return [name for name in HERO_NAMES if _normalize_text(name) in normalized_question]


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


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

    # Nếu user nhắc đích danh tướng thì ưu tiên lấy đúng tướng đó trước.
    for hero_name in _extract_mentioned_heroes(question):
        hero_result = collection.query(query_texts=[question], n_results=RERANKER_CANDIDATE_K, where={"name": hero_name})
        hero_docs = hero_result.get("documents", [[]])
        if hero_docs and hero_docs[0]:
            docs.extend(hero_docs[0])

    candidate_k = max(RERANKER_CANDIDATE_K, top_k * 2)
    semantic_result = collection.query(query_texts=[question], n_results=candidate_k)
    semantic_docs = semantic_result.get("documents", [[]])
    if semantic_docs and semantic_docs[0]:
        docs.extend(semantic_docs[0])

    candidates = _dedupe_preserve_order(docs)
    if not candidates:
        return []

    pairs = [(question, candidate) for candidate in candidates]
    scores = _get_reranker().predict(pairs)
    scored_candidates = sorted(zip(candidates, scores), key=lambda item: item[1], reverse=True)

    final_top_k = min(top_k, RERANKER_TOP_K) if top_k else RERANKER_TOP_K
    return [candidate for candidate, _ in scored_candidates[:final_top_k]]
