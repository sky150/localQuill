from langchain_ollama import OllamaEmbeddings

# Calls the local Ollama API to get the embedding function. 
def get_embedding_function_local():
    # nomic-embed-text as the first model
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings
