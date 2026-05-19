# BABY-AGI(`아기`): Layered Intelligence Architecture

---

## Architectural Philosophy

Modern humans navigate the contemporary world using a Paleolithic neurological blueprint. We do not constantly rewire our core brain (weights) for every new task; instead, we rely on external tools, adaptable memories, and highly efficient emotional heuristics.

This project aims to implement a conceptual model of AGI through a **Layered Scaffolding Architecture**:


### 1. Reasoning and Memorization (System 1: The Frozen Cortex)

The cognitive backbone. It does not learn in real-time. It operates purely on provided context and pre-trained logic.

- **L1 (Deterministic):** External tools and rigid logic (Calculator, Interpreter, Search). We offload precise tasks here, just as humans use smartphones.
- **L2 (Short-Term Memory):** Raw sensory context (VLM) and working memory (Prompts). Very accurate but volatile.
- **L3 (Long-Term Memory):** Lossy, abstracted episodic memory (Vector DB). Not every detail is saved; only 'events' tagged with high emotion.
- **L4 (Instincts & Reasoning):** **Frozen,** pre-trained base weights (Large Language Models).


### 2. Emotion Network (System 0: The Plastic Engine)

**The Core Engine:** The only continuously mutating component. It acts as an ultra-lightweight, plastic routing engine. It evaluates survival parameters, shifts attention, triggers memory addition/eviction, and controls System 1 via "Soft-Prompt Tokens" (e.g., `<ANXIOUS>`, `<CURIOUS>`).

- **Mathematical Intuition of 'Surprise' (RPE):** Emotion is mathematically driven by the mismatch between expectation and reality. Crucially, this learning mechanism is **Dual-Path**—it learns not only from the outside world but also from its own recursive thoughts.

$$Surprise = | Expected\_Outcome - Feedback (External\_Reality \cup Internal\_Critique) |$$

- **External (Social) Surprise:** If the system predicts the user will praise it but receives a penalty (Error/Scolding), the massive 'Surprise' spike triggers an immediate weight update in System 0.
- **Internal (Reflective) Surprise:** System 0 constantly monitors System 1's "Inner Monologue" (Internal CoT). If the system detects a logical fallacy, cognitive dissonance, or sycophancy (e.g., realizing *"I am just generating this to please the user, but it contradicts my stored facts and thoughts"*), this internal mismatch generates an intrinsic surprise spike. This allows the system to feel 'doubt' or 'epiphanies' recursively, evolving its weights even without user intervention.


### 3. The Manager Network (Sleep & Garbage Collection)

Intelligence needs downtime. The Manager Network is a background process that activates when System 0 detects a prolonged state of `<CALM>` (Idle time / "Sleep").

- **Memory Consolidation:** Scans L2 (Recent logs) and L3 (Vector DB). Merges repetitive observations into high-level concepts.
- **Active Forgetting (Garbage Collection):** Drops data points with an `Arousal_Score` near zero. If an event wasn't painful or joyful, it is deleted to lower entropy and save disk space.
  - The old but unimportant memories also should be removed. 


### 4. Safety & Containment: "The Playpen"

How do we control an evolving entity? We do not constrain its intelligence and emotion; we restrict its hands (L1).

- **Restricted Embodiment:** The AI exists in a digital sandbox. It may use Python's `eval()` but with strict namespace limitations (e.g., stripping `__builtins__`, `os`, `sys`).
- **One-Way Vision:** The VLM (Visual Observer) can *see* the screen but cannot *click* or *type* directly (no `pyautogui` access for the agent). It must politely ask the user or use a heavily guarded tool interface.
- **Self-Correction Loop (`Conscience`):** System 0 constantly monitors System 1's "Inner Monologue." If System 1 plans a malicious tool call, System 0 may intercept it via a negative arousal spike before execution.


### 5. Modular Interaction: Token-Mediated Interpretability

Unlike experimental models that attempt complex "weight-mixing" or high-dimensional gradient synchronization—which often lead to **"Computational Explosion"** and "Black-box behavior"—this project ensures a clean, decoupled interaction protocol.

- **Non-Invasive Interaction:** System 0 (Emotion) does not touch the weights of System 1 (Reasoning). Instead, communication is mediated through **Discrete Tokens**.
- **Human-Readable Traceability:** Because emotions are excreted as human-readable strings (e.g., `<ANXIOUS>`, `<CURIOUS>`), the developer can trace the exact "feeling" that triggered a specific "thought."
- **Zero-Latency Overhead:** Passing string-based tokens between modules is computationally near-zero compared to real-time neural weight updates. This allows the system to remain lightning-fast on local hardware (MBP M4 Pro) while maintaining deep, multi-layered logic.

---

## Test ENV

- MBP M4 Pro 24G RAM

## Setup

```sh
conda env create -f environment.yml   # NOTE : conda env update -f environment.yml (if required) 
conda activate baby_agi
```
