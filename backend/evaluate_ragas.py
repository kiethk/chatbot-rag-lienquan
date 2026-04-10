import json
import sys
from pathlib import Path

from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics._answer_relevance import answer_relevancy
from ragas.metrics._context_precision import context_precision
from ragas.metrics._faithfulness import faithfulness

from chat_service import generate_reply
from config import get_api_key
from question_processor import rewrite_question
from vector_store import ingest_data, retrieve_context


BASE_DIR = Path(__file__).resolve().parent
DATASET_FILE = BASE_DIR / "eval_dataset.json"
GITHUB_MODELS_BASE_URL = "https://models.inference.ai.azure.com"
LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def load_examples() -> list[dict]:
    with DATASET_FILE.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def build_evaluation_rows(examples: list[dict]) -> list[dict]:
    rows = []
    for example in examples:
        question = example["question"]
        reference = example["reference"]
        processed_question = rewrite_question(question, [])
        contexts = retrieve_context(processed_question)
        response = generate_reply(processed_question, [], "\n".join(contexts))
        rows.append(
            {
                "user_input": question,
                "response": response,
                "retrieved_contexts": contexts,
                "reference": reference,
            }
        )
    return rows


def main() -> None:
    ingest_data()
    examples = load_examples()
    rows = build_evaluation_rows(examples)
    dataset = Dataset.from_list(rows)

    api_key = get_api_key()
    if not api_key:
        raise ValueError("Missing API key. Set GITHUB_TOKEN or OPENAI_API_KEY in backend/.env")

    llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=LLM_MODEL,
            temperature=0,
            base_url=GITHUB_MODELS_BASE_URL,
            api_key=api_key,
        )
    )
    embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=GITHUB_MODELS_BASE_URL,
            api_key=api_key,
        )
    )

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=llm,
        embeddings=embeddings,
        show_progress=True,
    )

    print("\nAverage scores:")
    print(result.to_pandas().mean(numeric_only=True).to_string())
    print("\nPer-example results:")
    print(result.to_pandas().to_json(orient="records", force_ascii=False, indent=2))


if __name__ == "__main__":
    main()
