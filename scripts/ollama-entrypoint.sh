#!/bin/sh
# ollama-entrypoint.sh
# Starts the Ollama server and pulls models for the active quality_preset.
# Preset is read from QUALITY_PRESET env var (default: local_cpu).

set -e

echo ">>> Starting Ollama server..."
ollama serve &
OLLAMA_PID=$!

echo ">>> Waiting for Ollama API..."
until curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; do
  sleep 2
done
echo ">>> Ollama API is ready."

# Select model list based on preset
PRESET="${QUALITY_PRESET:-local_cpu}"
echo ">>> Active preset: $PRESET"

case "$PRESET" in
  local_gpu)
    MODELS="llama3.3:70b llama3.1:8b-instruct mxbai-embed-large"
    ;;
  local_cpu)
    MODELS="qwen2.5:7b-instruct-q4_K_M qwen2.5:1.5b-instruct mxbai-embed-large"
    ;;
  *)
    # Unknown or cloud preset — pull nothing (cloud uses Bedrock/OpenAI)
    echo ">>> Preset '$PRESET' does not require Ollama models. Skipping pulls."
    MODELS=""
    ;;
esac

for MODEL in $MODELS; do
  echo ">>> Checking model: $MODEL"
  # Search for the exact model name and tag
  if ollama list | grep -q "$MODEL"; then
    echo "    $MODEL already present, skipping pull."
  else
    echo "    Pulling $MODEL (this may take a while on first run)..."
    ollama pull "$MODEL"
    echo "    $MODEL ready."
  fi
done

echo ">>> All models ready. Ollama is healthy."
wait $OLLAMA_PID
