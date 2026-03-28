import os
import json
from dotenv import load_dotenv
import src.g_eval.retrieval_metrics as retrieval_metrics
from src.model_query import get_db, text_normalization, similarity_search


def retrieval_evaluation():
    with open("test/retrieval_test_questions.json", "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    eval_model = os.getenv("EVAL_MODEL", "all-MiniLM-L6-v2")
    chunk_size = os.getenv("EVAL_CHUNK_SIZE", "1500")
    eval_db_path = os.getenv("CHROMA_EVAL_PATH", "./test/chroma_eval")
    collection_name = os.getenv("COLLECTION_NAME", "essay")

    print(f"=== Starting Retrieval Evaluation ===")
    print(f"Embedding Model: {eval_model}")
    print(f"Chunk Size: {chunk_size}")
    print("=" * 37 + "\n")

    total_score = 0
    valid_tests = 0

    try:
        eval_db = get_db(chroma_path=eval_db_path, collection_name=collection_name)
    except Exception as e:
        print(f"Could not load test database at {eval_db_path}.")
        print(f"Error: {e}")
        print("Did you run generate_chroma.py pointing to the eval path?")
        return

    for i, tc in enumerate(test_cases):
        query = tc["query"]
        print(f"Test Case {i+1}/{len(test_cases)}: {query}")

        normalized_query = text_normalization(query)

        results = similarity_search(
            db=eval_db,
            user_text=normalized_query,
            collection_name=collection_name,
            top_k=3,
        )

        if isinstance(results, str):
            retrieved_chunks = []
        else:
            retrieved_chunks = [doc.page_content for doc, score in results]

        if not retrieved_chunks:
            print("No chunks retrieved for this query.")
            print("-" * 50)
            continue

        score, reason = retrieval_metrics.evaluate_retrieval(query, retrieved_chunks)
        total_score += score
        valid_tests += 1

        print(f"Relevancy Score: {score}")
        print(f"Reason: {reason}")
        print("-" * 50)

    if valid_tests > 0:
        average_score = total_score / valid_tests
        print(f"\nFinal Average Relevancy Score for {eval_model}: {average_score:.2f}")


if __name__ == "__main__":
    load_dotenv()
    retrieval_evaluation()
