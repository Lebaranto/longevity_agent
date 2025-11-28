# Longevity Agent PoC

This repository contains a proof-of-concept multi-agent pipeline for reasoning about longevity interventions. It wires a small world model (ontology-backed knowledge graph) into a LangGraph workflow with guardrailed intent detection, micro-reasoning, translation analysis, and lightweight experiment planning. A minimal Streamlit UI is provided for interactive exploration.

## What this PoC does
- Detects whether a user query is relevant to longevity/aging and extracts intervention names.
- Pulls structured facts from the ontology-backed `world_model.py` and computes formal translation scores.
- Adds an LLM-driven micro-reasoning step to bridge formal logic with pragmatic intuition.
- Generates cross-species translation analyses and proposes next-step experiments under budget/time constraints.
- Produces a Markdown report and renders results in a simple Streamlit dashboard.

## Project layout
- `ontology.yaml` – demo ontology (species, pathways, genes, interventions, scoring rules).
- `world_model.py` – Pydantic-powered loader and scoring logic for the ontology.
- `agent_system.py` – LangGraph workflow (intent → world model query → micro reasoning → translation → planning → report).
- `streamlit_ui.py` – Streamlit app for interactive runs.
- `main.py` – CLI entrypoint for single-query runs.
- `utils.py` – optional helpers (configuration, logging, persistence, caching).

## Running the agent
1. Install dependencies (example):
   ```bash
   pip install -r requirements.txt
   ```
   Ensure `OPENAI_API_KEY` is set in your environment.

2. CLI usage:
   ```bash
   python main.py "Analyze rapamycin for human longevity translation"
   ```

3. Streamlit UI:
   ```bash
   streamlit run streamlit_ui.py
   ```