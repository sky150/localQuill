# Writers Guide

This project uses [UV](https://github.com/astral-sh/uv) for Python environment and dependency management.

## Setup Instructions

### 1. Install UV (Windows)

Open PowerShell and run:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

After installation, restart your terminal.

### 2. Create a Virtual Environment

To create a new virtual environment with Python 3.12:

```sh
uv venv --python 3.12
```

To change the Python version and reseed:

```sh
uv venv --python 3.12 --seed
```

### 3. Activate the Virtual Environment

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Initialize the Project

In the project folder, run:

```sh
uv init
```

### 5. Add Dependencies

To add a package:

```sh
uv add PACKAGE
```

### 6. Run the Project

To run your Python code:

```sh
uv sync
uv run main.py
```
## Ollama

Download Ollama
Powershell
irm https://ollama.com/install.ps1 | iex

Run Ollama
ollama

## OpenWeb UI
Guide
https://docs.openwebui.com/getting-started/quick-start/

Update 
OpenWebUI
uv add open-webui

Run in seperate Powershell instance, not tied to the Terminal in VS-Code. Ollama should run before starting this to have access to the Port with the Local AI Models

$env:DATA_DIR="C:\open-webui\data"; uvx --python 3.12 open-webui@latest serve


Sequency Diagramm (MermaidLive):

```
sequenceDiagram
    Chainlit->>+Backend: User Text
    Backend->>+ChromaDB: RAG Similarity Search
    ChromaDB->>+Backend: Context Vectors
    Backend->>+Mini-LLM: Context and Prompt
    Mini-LLM->>+Chainlit: Response
```

![Mermaid Sequence Diagram](/.attachements/ "Mermaid SequenceDiagram")



Llama Index - LangChain. If we need it. (LlamaaIndex easier to start)

# Run Project

## Generate RAG DB (Chroma)
```sh
uv sync
uv run -m src.vector_db.generate_chroma
```

## Rest RAG DB 
Deletes all Chroma files. Allowing them to be regenerated with changed files and new chunk metaparameters.
```sh
uv sync
uv run -m src.vector_db.generate_chroma --reset
```

## Add Collections

### Mac / Linux
```sh
COLLECTION_NAME=essay DATA_PATH=./data/styles/essay uv run -m src.vector_db.generate_chroma

COLLECTION_NAME=fiction DATA_PATH=./data/styles/fiction uv run -m src.vector_db.generate_chroma
```
### Windows
```sh
$env:COLLECTION_NAME="essay"; $env:DATA_PATH="./data/styles/essay"; uv run -m src.vector_db.generate_chroma
$env:COLLECTION_NAME="fiction"; $env:DATA_PATH="./data/styles/fiction"; uv run -m src.vector_db.generate_chroma
```


## Run Main
```sh
uv sync
uv run -m main.py
```

## Run frontend (Chainlit)
### Mac / Linux
```sh
PYTHONPATH=. uv run chainlit run src/frontend/app.py
```

### Windows
```sh
$env:PYTHONPATH="."; uv run chainlit run src/frontend/app.py
```


To test the rag
### Mac / Linux
```sh
PYTHONPATH=. uv run python test/test_rag.py
```

### Windows
```sh
$env:PYTHONPATH="."; uv run python test/test_rag.py
```

## G-Eval
### Generate seperate DB for evaluation

```sh
COLLECTION_NAME=essay DATA_PATH=./data/eval CHROMA_PATH=./tests/chroma_eval uv run -m src.vector_db.generate_chroma
```

### Run Tests

#### Mac / Linux
```sh
PYTHONPATH=. uv run python tests/eval/run_retrieval_eval.py
PYTHONPATH=. uv run python tests/eval/run_generation_eval.py
```

### Windows
```sh
$env:PYTHONPATH="."; uv run python tests/eval/run_retrieval_eval.py
$env:PYTHONPATH="."; uv run python tests/eval/run_generation_eval.py
```

### Jupyter notebooks
```sh
uv run jupyter notebook
```
