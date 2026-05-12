from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import OllamaModel

import os
from dotenv import load_dotenv

load_dotenv()

ollama_model = OllamaModel(
    model=os.getenv("LOCAL_MODEL", "llama3.1:8b")
)  # os.getenv("LOCAL_MODEL", "llama3.1:8b")

precision_metric = ContextualPrecisionMetric(
    threshold=0.7, model=ollama_model, include_reason=True
)
recall_metric = ContextualRecallMetric(
    threshold=0.7, model=ollama_model, include_reason=True
)


# Evaluate the Retrieval step: how well the embedding model finds the right context from the PDFs


def evaluate_retrieval(user_query: str, retrieved_chunks: list, expected_concept: str):
    """
    Use Precision metric and recall metric to evaluate only retrieval
    """
    test_case = LLMTestCase(
        input=user_query,
        actual_output=expected_concept,
        expected_output=expected_concept,
        retrieval_context=retrieved_chunks,
    )

    precision_metric.measure(test_case)
    recall_metric.measure(test_case)

    return (
        precision_metric.score,
        precision_metric.reason,
        recall_metric.score,
        recall_metric.reason,
    )


def evaluate_document_hit(expected_document: str, retrieved_metadata_list: list):
    """
    Checks if the expected document is present in the metadata of the retrieved chunks.
    """
    if not expected_document or expected_document == "N/A":
        return False

    # Extract just the filenames from the retrieved metadata sources
    retrieved_filenames = [
        os.path.basename(meta.get("source", "")) for meta in retrieved_metadata_list
    ]

    return expected_document in retrieved_filenames


def debug_metadata(retrieved_metadata_list: list):
    """Run once to confirm the metadata key your eval DB uses."""
    for i, meta in enumerate(retrieved_metadata_list):
        print(
            f"  [chunk {i}] keys: {list(meta.keys())} | source: {meta.get('source', 'MISSING')}"
        )
