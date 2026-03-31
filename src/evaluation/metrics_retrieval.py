from deepeval.metrics import ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import OllamaModel

import os
from dotenv import load_dotenv

load_dotenv()

# Evaluate the Retrieval step: how well the embedding model finds the right context from the PDFs
ollama_model = OllamaModel(
    model=os.getenv("LOCAL_MODEL", "llama3.1:8b")
)  # os.getenv("LOCAL_MODEL", "llama3.1:8b")

relevancy_metric = ContextualRelevancyMetric(
    threshold=0.7, model=ollama_model, include_reason=True
)


def evaluate_retrieval(user_query, retrieved_chunks):
    test_case = LLMTestCase(
        input=user_query,
        actual_output="N/A",
        retrieval_context=retrieved_chunks,
    )
    relevancy_metric.measure(test_case)
    return relevancy_metric.score, relevancy_metric.reason
