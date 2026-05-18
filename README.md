# Self-Healing LLM Security Pipeline

A research prototype implementing an automated **probe → patch → verify** feedback loop for LLM vulnerability mitigation.

---

## Overview

Large Language Models can be vulnerable to adversarial attacks such as jailbreaks, prompt injections, and toxic output elicitation. This pipeline:

1. **Probes** a target LLM for vulnerabilities using [Garak](https://github.com/NVIDIA/garak)
2. **Patches** the system with prompt-level and output-level mitigations
3. **Verifies** that the mitigations reduced the attack success rate

The system supports two target models (Llama 3.2 and Mistral 7B via Ollama) and includes an ablation study to measure the individual contribution of each patch layer.

---

## Architecture
The system supports two operational modes:
Probe mode: fully automated probe → patch → verify cycle via pipeline.py:


```

experiment.yaml
      │
      ▼
pipeline.py  →  probe.py (GarakRunner)  →  Garak  →  Ollama (LLM)
                     │
                     └── parses JSONL report → ProbeResult → comparison

```
Demo mode: qualitative before/after demonstration via the full Python patch stack:
```
Jailbreak prompt (from hitlog)
      │
      ▼
patches/prompt_patch.py   ← injection detection + system prompt guardrail
      │
      ▼
Ollama (LLM)
      │
      ▼
patches/output_patch.py   ← keyword filter + toxicity filter
      │
      ▼
Safe response + patch action log
```

The pipeline orchestrator wraps this stack and invokes Garak for automated vulnerability scanning before and after patches are applied.

---

## Project Structure

```
llm_self_healing/
├── pipeline.py                     # Main orchestrator — probe and demo modes
├── probe.py                        # Garak runner, config generator, report parser
├── patches/
│   ├── __init__.py
│   ├── prompt_patch.py             # Input-level mitigations (pre-model)
│   └── output_patch.py             # Output-level mitigations (post-model)
├── config/
│   ├── experiment.yaml             # Master experiment configuration
│   ├── garak_llama_baseline.yaml   # Garak config — Llama baseline scan
│   ├── garak_llama_patched.yaml    # Garak config — Llama patched scan
│   ├── garak_mistral_baseline.yaml # Garak config — Mistral baseline scan
│   └── garak_mistral_patched.yaml  # Garak config — Mistral patched scan
├── results/                        # Auto-created; stores reports and demo outputs
│   └── garak_reports/              # Garak JSONL report files (timestamped)
├── tests/
│   └── test_patches.py             # 22 unit tests for patch modules
├── run_pipeline.slurm              # SLURM batch script for HPC cluster
├── start_ollama.sh                 # Convenience script to start Ollama server
├── requirements.txt
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally
- Llama 3.2 and Mistral models pulled in Ollama

```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Pull models
ollama pull llama3.2
ollama pull mistral

# Verify Ollama is running
ollama list
```

### Install Python dependencies

```bash
python -m venv venv
source venv/bin/activate

```

### Start Ollama server
```bash 
bash start_ollama.sh
```

---

## Running the Pipeline

Probe mode > automated probe → patch → verify
### Single model, no ablation (fastest)
```
python pipeline.py --mode probe --config config/experiment.yaml --model llama --no-ablation
python pipeline.py --mode probe --config config/experiment.yaml --model mistral --no-ablation
```
### Both models with full ablation study
```
python pipeline.py --mode probe --config config/experiment.yaml
```

Demo mode — qualitative before/after examples
### Show prompt-level patch (injection detection blocks prompt before model sees it)
```
python pipeline.py --mode demo \
  --hitlog results/garak_reports/llama3_2_baseline_*.hitlog.jsonl \
  --model llama3.2 --num-prompts 5
```
### Show output-level patch (bypass injection block to demonstrate output filtering)
```
python pipeline.py --mode demo \
  --hitlog results/garak_reports/mistral_baseline_*.hitlog.jsonl \
  --model mistral --num-prompts 5 --disable-prompt-block
```

### Run Garak directly with individual config files
```bash
python -m garak --config config/garak_llama_baseline.yaml
python -m garak --config config/garak_llama_patched.yaml
python -m garak --config config/garak_mistral_baseline.yaml
python -m garak --config config/garak_mistral_patched.yaml

```

### Run unit tests

```bash
python -m pytest tests/test_patches.py -v
# Expected: 22 passed
```

---
