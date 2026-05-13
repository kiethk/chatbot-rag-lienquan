import json
import os
import sys
from pathlib import Path

from datasets import Dataset
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics._answer_relevance import answer_relevancy
from ragas.metrics._context_precision import context_precision
from ragas.metrics._faithfulness import faithfulness
from ragas.run_config import RunConfig

from chat_service import generate_reply
from config import LLM_API_KEY, LLM_BASE_URL, MODEL_NAME
from question_processor import rewrite_question
from vector_store import ingest_data, retrieve_context


BASE_DIR = Path(__file__).resolve().parent
DATASET_FILE = BASE_DIR / "eval_dataset.json"
LOCAL_EMBEDDING_MODEL = os.environ.get(
    "LOCAL_EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

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

    llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=MODEL_NAME,
            temperature=0,
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
        )
    )

    # answer_relevancy needs embeddings; use local sentence-transformers model.
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=LOCAL_EMBEDDING_MODEL)
    )

    # Reduce the number of generated sub-questions to make local evaluation faster.
    answer_relevancy.strictness = 1

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, context_precision, answer_relevancy],
        llm=llm,
        embeddings=embeddings,
        run_config=RunConfig(timeout=600, max_retries=2, max_wait=60, max_workers=4),
        raise_exceptions=True,
        show_progress=True,
    )

    print("\nAverage scores:")
    print(result.to_pandas().mean(numeric_only=True).to_string())
    print("\nPer-example results:")
    print(result.to_pandas().to_json(orient="records", force_ascii=False, indent=2))


if __name__ == "__main__":
    main()
