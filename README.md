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
uv run main.py
```