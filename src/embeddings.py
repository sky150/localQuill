from langchain_ollama import OllamaEmbeddings

def get_embedding_function():
    # nomic-embed-text as the first model
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings
