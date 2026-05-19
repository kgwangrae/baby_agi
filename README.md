# BABY-AGI(`아기`): Layered Intelligence Architecture

----

## Architectural Philosophy

Modern humans navigate the contemporary world using a Paleolithic neurological blueprint. We do not constantly rewire our core brain (weights) for every new task; instead, we rely on external tools, adaptable memories, and highly efficient emotional heuristics.

This project aims to implement a conceptual model of AGI through a **Layered Scaffolding Architecture**:

### Reasoning and Memorization (System 1)

- **L1 (Deterministic):** External tools and rigid logic (Calculator, Interpreter, Search, Taking notes - as we do daily).
- **L2 (Short-Term Memory):** Raw sensory context (VLM) and working memory (Prompts).
- **L3 (Long-Term Memory):** Lossy, abstracted episodic memory (Vector DB).
- **L4 (Instincts & Reasoning):** **Frozen,** pre-trained base weights (Large Language Models).

### Emotion Network (System 0)

- **The Core Engine:** The only continuously mutating component. It acts as an ultra-lightweight, plastic routing engine. It evaluates survival parameters, shifts attention, triggers memory addition, consolidation and eviction, and controls the System 1 via soft-prompt tokens.
- **Definition of 'Surprise'**
  - The expectation from System 1 meets external feedback (i.e., input prompts from the user), or coherent with recursive re-thinking (i.e., secondary fact-check with tools, comparison with conflicting hypothesis)

----

## Test ENV

- MBP M4 Pro 24G RAM

## Setup

```sh
conda env create -f environment.yml   # NOTE : conda env update -f environment.yml  (if required) 
conda activate baby_agi
```
