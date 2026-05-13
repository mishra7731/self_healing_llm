#!/bin/bash
module load ollama/0.6.5
export OLLAMA_MODELS=/scratch/general/vast/u1457424/ollama_models
ollama serve &
sleep 4
curl http://localhost:11434
echo ""
echo "Ollama is ready. Models available:"
ollama list
