import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model_query import query_rag
from src.embeddings import get_embedding_function_local
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma")

TEST_TEXT = """I am writing this comment to complain about the worrying increase..."""  # your full text


def test_connection():
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function_local()
    )
    count = db.get()
    print(f"Connected! Documents in DB: {len(count['ids'])}")


def test_retrieval():
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function_local()
    )
    results = db.similarity_search_with_score(TEST_TEXT, k=3)
    for doc, score in results:
        print(f"Score: {score:.3f}")
        print(f"Source: {doc.metadata.get('source', '?')}")
        print(f"Content: {doc.page_content[:150]}...")


def test_full_pipeline():
    response = query_rag(TEST_TEXT)
    print(response)


if __name__ == "__main__":
    test_connection()
    test_retrieval()
    test_full_pipeline()
