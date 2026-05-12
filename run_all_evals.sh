#!/bin/bash

# Array of the embedding models you want to test
MODELS=(
    "mxbai-embed-large:latest"
    "dengcao/Qwen3-Embedding-0.6B:Q8_0"
    "embeddinggemma:latest"
    "nomic-embed-text:latest"
    "locusai/all-minilm-l6-v2:latest"
)

# Number of seconds to pause between runs to let your Mac cool down
SLEEP_TIME=20

echo "=== Starting automated retrieval evaluation for ${#MODELS[@]} models ==="

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "=========================================================="
    echo "▶ Evaluating Model: $MODEL"
    echo "=========================================================="

    # 1. Wake up the model using the embeddings API (safest way for embedding models)
    echo "Waking up $MODEL..."
    curl -s -X POST http://localhost:11434/api/embeddings -d "{\"model\": \"$MODEL\", \"prompt\": \"test\"}" > /dev/null
    
    # 2. Run the evaluation script
    echo "Running evaluation..."
    EVAL_MODEL="$MODEL" PYTHONPATH=. caffeinate -d uv run python tests/eval/run_retrieval_eval.py
    
    # 3. Check if this is the last model so we don't sleep unnecessarily at the end
    if [ "$MODEL" != "${MODELS[${#MODELS[@]}-1]}" ]; then
        echo "----------------------------------------------------------"
        echo "Finished $MODEL. Pausing for $SLEEP_TIME seconds to let Ollama breathe..."
        sleep $SLEEP_TIME
    fi
done

echo ""
echo "✅ All model evaluations completed successfully!"
