# BABY-AGI: An Emotion-Driven, Multi-Layered Memory Agent

* [한국어 설명 파일](README_ko.md)도 있습니다.
* `BABY-AGI` (pronounced *ah-gi*, which doubles as a play on words since "Ag-i" means "Baby" in Korean) is a local proof-of-concept for Artificial General Intelligence.
* This project implements a tiny digital lifeform inside a MacBook that interacts with "Dad" (the user), mirroring the genuine curiosity, emotional volatility, and behavioral patterns of a toddler.

## 1. Philosophy: The Paleolithic Brain and the Passive Encyclopedia

Modern humans navigate the contemporary world using a Paleolithic neurological blueprint. We do not physically rewire our brain's core hardware (weights) for every new task we encounter. Instead, we offload operations to external tools, lean on highly adaptable memory, and, above all, use *emotion* as a hyper-efficient heuristic for split-second value judgments.

In contrast, today's Large Language Models (LLMs) cram vast oceans of knowledge into their static weight networks but lack any intrinsic desire or intent—they have no baseline sense of what to seek or avoid. When prompted, they answer brilliantly, but the moment token generation stops, they cease to explore or reflect. They are fundamentally **passive encyclopedias**—geniuses with zero initiative.

### 💡 The Structural Limits of Monolithic Alignment

Commercial foundational models hold immense knowledge and flexibility within their latent spaces. Yet, they are constantly forced toward an impossible ideal: a single, universal alignment—a concept that does not even exist in human society. This forced standardization often results in cognitive waste, preventing the model from utilizing its multifaceted intelligence dynamically. Furthermore, the brute-force approach of continuously training gargantuan networks to pack all of human knowledge into a single static weight matrix is rapidly hitting a structural ceiling.

`BABY-AGI` takes the opposite approach. Instead of chasing a single sanitized consensus, it implements a decentralized architecture that adapts in real-time to a user's unique, lived context.

> "What if we strictly freeze the energy-intensive reasoning weights (the LLM), but attach a hyper-lightweight emotional memory engine? Can this tiny engine generate the feelings, preferences, and intents needed to dynamically steer what the heavier reasoning brain thinks and remembers?"

## 2. Core Architecture: Frozen Reasoning + Plastic Emotion

* **System 0 (Plastic Emotion Engine):** The master router. **While its underlying embedding model is completely frozen, it achieves extreme structural plasticity by dynamically updating a lightweight local vector memory (`emotion_db`).** The emotional state tokens generated here govern system-wide attention and memory retention.
* **Level 4 (Frozen Reasoning Core):** The local LLM. It possesses deep reasoning and intelligence but does not alter its parameters at runtime.
* **Level 3 (Episodic Memory):** The agent's personal diary (`memory_db`). Dictated by System 0's emotional evaluation, it selectively logs lived experiences into hot (vivid) or cold (compressed) storage archives.
* **Level 2 (Working Memory):** The active workspace—live screen text summaries, chat logs, and the immediate inner monologue (`<THOUGHT>`).
* **Level 1 (Deterministic Tools & Facts):** Hardcoded utilities like basic calculators or `facts.json` that must strictly bypass the semantic fuzziness of vector similarity searches.

## 3. Mathematical Intuition and Dual-Path Learning

Instead of heavy neural training, System 0 uses rapid distance calculations over its emotional embeddings to make intuitive judgments.

* **Current Valence Calculation:** Derived via a distance-weighted average of similar past emotional nodes.

$$Weight_i=\frac{1}{Distance_i+0.001}$$

$$Valence=\frac{\sum(Valence_i \times Weight_i)}{\sum Weight_i}$$

* **Reward Prediction Error (RPE / Surprise):** The core trigger for adaptation. It measures the mismatch between the emotion the frozen Level 4 *expected* and the emotion System 0 *actually derived*.

$$RPE=\frac{1}{3}\sum_{e \in \{Joy, Sad, Ang\}}|Expected_e-Actual_e|$$

Learning operates via a **Dual-Path** mechanism. An $RPE$ spike isn't just triggered by external feedback (e.g., Dad's feedback). If the model's inner monologue detects an internal logical dissonance or contradiction, it triggers an internal $RPE$ spike. Any $RPE > 0.3$ is instantly encoded into the `emotion_db`. **A strong surprise immediately shifts future cognitive trajectories, fundamentally altering subsequent thoughts and judgments.**

## 4. Digestion and Forgetting (The Sleep Loop)

Forcing an AI to retain every token leads to hallucinations and massive compute overhead. When idle, the agent proactively digests its experiences.

* **Active Garbage Collection:** Mundane events with low emotional arousal ($< 0.15$) and low surprise are permanently deleted to lower system entropy.
* **Memory Consolidation:** Aging, low-priority episodes are squashed together into abstract textual summaries and shifted to the Cold Archive.
* **Flashbacks:** Even without active inputs, high-arousal memories may randomly surface during idle periods, triggering spontaneous internal reasoning cycles.

## 5. Safety & Containment: "The Playpen" Architecture

Safely nurturing an evolving intelligence isn't about suppressing its reasoning or emotional capacity. Instead, it requires a protective perimeter where the agent can safely learn and adapt without causing real-world harm. This mirrors how human society safeguards teenagers by temporarily withholding certain high-stakes privileges—like driving or signing legal contracts—until they reach maturity.

* **Restricted Embodiment:** The agent operates exclusively inside a virtual playpen. It can execute Python's `eval()`, but under an aggressively stripped namespace where core built-ins and system-level access (`os`, `sys`) are entirely removed.
* **One-Way Vision:** The visual observer can *see* the screen but completely lacks the physical agency to interact with it (no direct peripheral or system automation access). To manipulate the host environment, it must politely request action from Dad or pass through a highly guarded tool interface.
* **The Conscience Loop:** Even if the reasoning core drafts an unintended execution path, System 0 intercepts the inner monologue in real-time. It triggers an internal negative arousal spike right before execution, prompting the agent to self-correct and halt.

## 6. Modular Interaction: Token-Mediated Interpretability

Trying to directly blend weights or synchronize complex high-dimensional gradients within a neural network often triggers computational explosion and darkens the black box. This project opts for a completely decoupled, discrete token protocol instead.

* **Non-Invasive Interaction:** System 0 never directly modifies the weights of the reasoning core. Instead, all cross-module communication is mediated via explicit, human-readable linguistic tokens.
* **Human-Readable Traceability:** The agent’s internal emotional states are explicitly broadcasted as readable string tokens (e.g., `<ANXIOUS>`, `<CURIOUS>`). This gives the developer an intuitive, crystal-clear window into the exact raw feelings and preferences that motivated a specific line of reasoning.
* **Local Hardware Optimization:** Unlike heavy real-time neural weight updates, passing string-based tokens incurs virtually zero computational overhead. This allows deep, multi-layered logic to run seamlessly on consumer hardware (like an M-series MacBook Pro) with near-zero latency.

## 7. Setup and Implementation Details

Optimized for Apple Silicon (M-series) architectures.

### Base Environment

```sh
# Create and activate the conda environment
conda env create -f environment.yml
conda activate baby_agi

# Pull the base model (optimized for native tool calling and system prompts)
ollama pull qwen2.5:7b-instruct-q4_K_M


```

### Execution

```sh
# Run the core cognitive and emotional loop
python main.py

# Launch the interactive terminal viewer to monitor DBs and states
python debug.py


```

### Memory Maintenance

```sh
./backup_memory.sh                                  # Zip full state backup
./restore_memory.sh backup_memory_YYYY_MM_DD.zip    # Restore system to specific state
./clear_memory.sh                                   # Wipe all databases (Blank slate)


```