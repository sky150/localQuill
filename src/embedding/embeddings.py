import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings


load_dotenv()


def get_embedding_function(logger=None):
    provider = os.getenv("EMBEDDING_PROVIDER", "huggingface")
    if logger:
        logger.info(f"Embedding provider: {provider}")
        logger.info(f"Embedding model: {os.getenv('EMBEDDING_MODEL')}")

    if provider == "ollama":
        return OllamaEmbeddings(
            model="nomic_embed-text",      #os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
            base_url="http://127.0.0.1:11434"
        )

    elif provider == "huggingface":
        return HuggingFaceEmbeddings(
            model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        )

    else:
        raise ValueError(f"Unknown embedding provider: {provider}")
    

# def get_chunk_size():
#     model = os.getenv("EMBEDDING_MODEL", "")

#     if "mpnet" in model:
#         return 350
#     elif "MiniLM" in model:
#         return 300
#     elif "nomic" in model:
#         return 250
#     else:
#         return 300
    

def chunk_user_prompt(text: str, chunk_size: int, chunk_overlap: int):
    words = text.split()
    chunks = []

    step = chunk_size - chunk_overlap

    for i in range(0, len(words), step):
        if len(chunk) < chunk_size * 0.5:
            break # Mini chunk which would falsify similarity search
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))

    return chunks