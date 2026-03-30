import os
import json
import platform
import datetime
import psutil
from dotenv import load_dotenv

load_dotenv()

EVAL_CONFIG = {
    "embedding_model": os.getenv("EVAL_MODEL", "all-MiniLM-L6-v2"),
    "chunk_size": int(os.getenv("EVAL_CHUNK_SIZE", 1500)),
    "chunk_overlap": int(os.getenv("EVAL_CHUNK_OVERLAP", 200)),
    "top_k": int(os.getenv("TOP_K", 3)),
    "llm_model": os.getenv("LOCAL_MODEL", "llama3.1:8b"),
    "judge_model": "llama3.1:8b",
    "eval_db_path": os.getenv("CHROMA_EVAL_PATH", "./tests/chroma_eval"),
    "collection_name": os.getenv("COLLECTION_NAME", "essay"),
    "results_path": "reports/eval_results.jsonl",
}


def get_eval_results(
    eval_type: str, duration: float, average_score: float, per_test_results: list
):
    ram_gb = round(psutil.virtual_memory().total / 1e9, 1)

    return {
        "timestamp": datetime.datetime.now().isoformat(),
        "eval_type": eval_type,
        "embedding_model": EVAL_CONFIG["embedding_model"],
        "llm_model": EVAL_CONFIG["llm_model"],
        "judge_model": EVAL_CONFIG["judge_model"],
        "chunk_size": EVAL_CONFIG["chunk_size"],
        "chunk_overlap": EVAL_CONFIG["chunk_overlap"],
        "top_k": EVAL_CONFIG["top_k"],
        "device_name": platform.node(),
        "ram_gb": ram_gb,
        "duration_seconds": round(duration, 2),
        "average_score": round(average_score, 4),
        "per_test": per_test_results,
    }


def save_eval_record(record):
    results_path = "reports/"
    os.makedirs(results_path, exist_ok=True)

    with open(results_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(f"\nResults saved to {results_path}")
