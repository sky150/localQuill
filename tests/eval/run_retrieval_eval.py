import os
import json
import time
from dotenv import load_dotenv
import src.evaluation.metrics_retrieval as metrics_retrieval
from src.model_query import get_db, text_normalization, similarity_search
from eval_config import EVAL_CONFIG, get_eval_results, save_eval_record


def retrieval_evaluation():
    with open("tests/eval/retrieval_test_questions.json", "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    print(f"=== Starting Retrieval Evaluation ===")
    print(f"Embedding Model: {EVAL_CONFIG['embedding_model']}")
    print(f"Chunk Size: {EVAL_CONFIG['chunk_size']}")
    print("=" * 36 + "\n")

    try:
        eval_db = get_db(
            chroma_path=EVAL_CONFIG["eval_db_path"],
            collection_name=EVAL_CONFIG["collection_name"],
        )
    except Exception as e:
        print(f"Could not load test database. Error: {e}")
        return

    results_log = []
    run_start = time.time()

    for i, tc in enumerate(test_cases):
        query = tc["query"]
        print(f"Test Case {i+1}/{len(test_cases)}: {query}")

        normalized_query = text_normalization(query)

        results = similarity_search(
            db=eval_db,
            user_text=normalized_query,
            collection_name=EVAL_CONFIG["collection_name"],
            top_k=EVAL_CONFIG["top_k"],
        )

        if isinstance(results, str):
            retrieved_chunks = []
        else:
            retrieved_chunks = [doc.page_content for doc, score in results]

        if not retrieved_chunks:
            print("No chunks retrieved for this query.")
            print("-" * 36)
            continue

        score, reason = metrics_retrieval.evaluate_retrieval(query, retrieved_chunks)

        print(f"Relevancy Score: {score}")
        print(f"Reason: {reason}")
        print("-" * 50)

        results_log.append(
            {
                "query": query,
                "expected_document": tc.get("expected_document", "N/A"),
                "score": score,
                "reason": reason,
            }
        )

    duration = time.time() - run_start

    if results_log:
        average_score = sum(r["score"] for r in results_log) / len(results_log)
        print(f"\nAverage relevancy score: {average_score:.2f}")

        record = get_eval_results("retrieval", duration, average_score, results_log)
        save_eval_record(record)


if __name__ == "__main__":
    load_dotenv()
    retrieval_evaluation()
