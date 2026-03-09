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
    UI->>+PromptAPI: Prompt
    PromptAPI->>+Rag: Query for context
    Rag->>+PromptAPI: Related Chunks
    PromptAPI->>+LocalLLM: Context and Prompt
    LocalLLM->>+UI: Response
```

![Mermaid Sequence Diagram](/.attachements/SequenceDiagram.png "Mermaid SequenceDiagram")


Llama Index - LangChain. If we need it. (LlamaaIndex easier to start)

# Run Project

## Generate RAG DB (Chroma)
```sh
uv sync
uv run -m src.vector_db.generate_chroma
```

## Rest RAG DB 
*Not yet functional*
```sh
uv sync
uv run -m src.vector_db.generate_chroma --reset
```

## Run Main
```sh
uv sync
uv run -m main.py
```