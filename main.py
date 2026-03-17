from langchain_ollama import OllamaLLM
import time

def main():
    start = time.time()
    model = OllamaLLM(
        model="qwen3.5:4b",
        base_url="http://127.0.0.1:11434",
        num_predict=300,
        temperature=0.3,
    )

    print(model.invoke("Say hello."))
    end = time.time()
    print(f"Response time: {end - start:.2f} seconds")


if __name__ == "__main__":
    main()
