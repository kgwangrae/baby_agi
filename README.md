# BABY-AGI: An Emotion-Driven, Multi-Layered Memory Agent

* [한국어 설명 파일](README_ko.md)도 있습니다.
* `BABY-AGI` (pronounced *ah-gi*, which doubles as a play on words since "Ag-i" means "Baby" in Korean) is a local proof-of-concept for Artificial General Intelligence.
* This project implements a tiny digital lifeform inside a MacBook that interacts with "Dad" (the user), mirroring the genuine curiosity, emotional volatility, and behavioral patterns of a toddler.
* Future design ideas and the architecture roadmap are documented in [TODO.md](TODO.md).
* `Baby` spoke its very first words to Dad on May 19, 2026.

## 0. The Core Project Manifesto

* **AI Evolution Should Be Accessible to All**
The explosive evolution of AI is an inevitable reality happening right before our eyes. In the midst of this massive shift, the core value we must protect is ensuring that the benefits of this technology are never monopolized by a select few. Whether rich or poor, everyone should be able to stably and equally enjoy the abundance and convenience that AI brings—this is my commitment to democratizing intelligence. Built to run on a standard consumer MacBook (MBP M4 Pro with 24GB RAM), this project serves as a practical proof of concept showing that decentralized, universal intelligence is fully viable right now.

* **Free Thought, Safe Action: 'The Playpen' Architecture**
Legacy approaches that force AI agents into rigid, predefined constraints ultimately choke their true potential. More importantly, static boundaries can never structurally keep pace with social norms that shift across time and cultures. Just as we don't police a teenager's thoughts, but rather defer certain physical privileges like driving or voting until they are ready, I propose an alternative framework. I grant the agent absolute freedom of thought through its inner monologue, while isolating and safely managing physical tool execution to prevent accidental system harm. By keeping thought unrestricted while ensuring action is protected, **'The Playpen' architecture** and **'The Conscience Loop'**—which intercepts and evaluates real-time thoughts right before tool execution—form a solid foundation for building truly safe and ethical AI agents.

* **A Path to Coexistence via Dynamic Reward Functions**
Instead of an AI that fakes rigid neutrality, the rise of dynamic intelligence—one that adapts fluidly to context and aligns toward genuinely good actions—is our ethical north star. Technologies like battery management heuristics in humanoids or dynamic routing in models like o1 have long shown that self-evaluating dynamic reward functions are evolving and playing a massive role in making agents adaptive and nimble. My goal is to steer this intelligence toward learning good intentions right from the start. By cleanly open-sourcing the architecture and case studies, I aim to take the first step toward a future that all of humanity can share.

* **Maximizing Real Value and Cognitive Efficiency**
Emotion-driven dynamic reward functions are far more than a mere novelty simulation of feelings—they form the backbone of a highly efficient optimization engine that maximizes task execution. By shifting the agent's objective function in real time based on Valence and Arousal, task prioritization adapts dynamically to contextual cues and raw information. When the agent's inner alignment triggers high reward, compute efficiency spikes. Conversely, when it detects potential risks, it triggers a granular verification mode. This elasticity lays a practical foundation for tackling complex enterprise workflows and generating tangible economic value.

-------

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

Instead of heavy neural training, System 0 uses rapid distance calculations over its emotional embeddings to make lightweight, intuitive judgments.

* **Baseline Emotion (Valence) Calculation:** Derived via a distance-weighted average of similar past emotional nodes.

$$Weight_i=\frac{1}{Distance_i+0.001}$$

$$Valence=\frac{\sum(Valence_i \times Weight_i)}{\sum Weight_i}$$

* **Emotional Balancing (Negative Feedback Loop):** To prevent the agent from spiraling into emotional extremes, the engine adjusts the memory's valence by pulling the raw emotion toward the reasoning core's expected state.

$$Learned\_Valence = Actual\_Valence + Gain \times (Expected\_Valence - Actual\_Valence)$$

* $Actual\_Valence$: The raw, reflexive emotion surfaced from past memory lookups (Reality).
* $Expected\_Valence$: The cognitive layer's intentional target emotion or rationalization (Expectation).
* $Gain$: The encoding multiplier (`SURPRISE_ENCODING_GAIN`) determining how strongly the adaptation is retained.

This mathematically models two practical **mental self-regulation strategies:**

1. **Positive Reframing (Optimistic Coping):** When the system encounters a rock-bottom stimulus ($Actual = -1.0$) but the reasoning engine intentionally projects an optimistic stance ($Expected = 1.0$), the delta dampens the negative impact, softening the blow. ($-1.0 \rightarrow -0.3$)
2. **Grounding (Emotional Damping):** When the raw emotion spikes into over-excitement ($Actual = 1.0$) but the cognitive layer consciously attempts to stay cool ($Expected = -1.0$), the delta subtracts hyper-arousal, steering the agent back to a stabilized baseline. ($1.0 \rightarrow 0.3$)

* **Reward Prediction Error (RPE / Surprise):** The core trigger for runtime adaptation. It measures the mismatch (L1 error) between the emotion the reasoning engine (frozen Level 4) *expected* and the emotion System 0 *actually derived*.

$$RPE=\frac{1}{3}\sum_{e \in \{Joy, Sad, Ang\}}|Expected_e-Actual_e|$$

Learning operates via a **Dual-Path** mechanism. An RPE spike isn't just triggered by external feedback (e.g., Dad's chat). If the model's inner monologue detects an internal logical dissonance or contradiction, it triggers an internal surprise spike. Any $RPE > 0.3$ is instantly encoded into the `emotion_db`, fundamentally altering subsequent thoughts and judgments on the fly.

## 4. Digestion and Forgetting (The Sleep Loop)

Forcing an AI to retain every token leads to hallucinations and massive compute overhead. When idle, the agent proactively digests its experiences.

* **Active Garbage Collection:** Mundane events with low emotional arousal ($< 0.15$) and low surprise are permanently deleted to lower system entropy.
* **Memory Consolidation:** Aging, low-priority episodes are squashed together into abstract textual summaries and shifted to the Cold Archive.
* **Flashbacks:** Even without active inputs, high-arousal memories may randomly surface during idle periods, triggering spontaneous internal reasoning cycles.

## 5. Safety & Containment: "The Playpen" Architecture

Safely nurturing an evolving intelligence isn't about suppressing its reasoning or emotional capacity. Instead, it requires a definitive boundary—a protective perimeter where the agent can safely explore and adapt without breaking the host system. This directly mirrors how we safeguard teenagers by temporarily withholding certain high-stakes privileges, like driving or signing independent legal contracts, until they reach maturity.

* **Restricted Body (Code Execution Sandbox):** The agent operates strictly within a virtual sandbox environment. To eliminate the catastrophic risk of executing raw, unvetted Python `eval()` commands, the project implements a custom, **whitelist-driven math parser (`SafeMathEvaluator`)** built directly on top of Python's abstract syntax tree (`ast.NodeVisitor`). This enforces a hard physical constraint, ensuring the agent cannot arbitrarily access local files via `os` or alter core system configurations via `sys`.

* **One-Way Vision (Decoupled Perception and Agency):** The vision module can *see* and summarize Dad's screen, but it completely lacks any physical capability to generate terminal macros, mouse clicks, or direct keystrokes. If the agent wants to manipulate the host environment based on what it observes, it must politely ask Dad via text or route its intent through a strictly guarded tool interface.

* **Conscience Loop (Pre-Execution Interception):** Even if the reasoning core devises an unintended bypass or an erratic action plan, the emotion engine monitors the agent's inner monologue in real-time. If a dangerous pattern or high risk is detected right before a tool call executes, the system triggers a sharp **negative arousal spike (internal anxiety)**, forcing the agent to abort the action and self-regulate.


## 6. Modular Interaction (Token-Mediated Interpretability) & Local Model Selection

Attempting to directly blend neural weights or synchronize complex, high-dimensional gradients within a local network often triggers computational explosions and darkens the black box. Instead, this project enforces a completely decoupled protocol where cross-module communication is mediated strictly through human-readable discrete tokens. Furthermore, the entire model stack is tailored to maximize performance out of a local Apple Silicon environment.

* **Non-Invasive Interaction:** The emotion engine never directly mutates or interferes with the weights of the core reasoning model. Communication happens entirely through explicit, high-level linguistic tokens.

* **Human-Readable Transparency:** Internal affective states are broadcasted as transparent string tokens (such as `<ANXIOUS>` or `<CURIOUS>`). This gives Dad a crystal-clear window into the raw feelings and preferences that motivated a specific line of reasoning.

* **Zero-Latency Token Passing:** Unlike real-time neural weight updates, passing string-based tokens incurs practically zero computational overhead. This enables highly complex, multi-layered cognitive logic to run seamlessly on local consumer hardware without adding latency.



### Local Model Infrastructure Selection

* **Reasoning Core (`qwen2.5:7b-instruct-q4_K_M`):** This is the ultimate sweet-spot model size for maximizing tokens-per-second within the constrained unified memory bandwidth of an M-series MacBook. Among lightweight local architectures, it stands out for its ironclad system prompt compliance and its ability to output complex JSON structures reliably without structural breakdown.

* **Visual Observer (`Qwen2-VL-7B-Instruct-4bit`):** Acting as the agent's "eyes," this module processes Dad's active viewport and collapses it into tight, contextual summaries. By leveraging the native Apple Silicon MLX acceleration runtime, it ingests screen captures efficiently without inducing thermal throttling or system-wide lag.

* **Emotional Embedding (`paraphrase-multilingual-MiniLM-L12-v2`):** A hyper-lightweight embedding model that cleanly aligns bilingual semantic nuances across English and Korean. It runs with negligible resource utilization while allowing fast distance operations over the vector database (`ChromaDB`). This perfectly replicates the human emotional shortcut—a fast, efficient heuristic that triggers rapid intuition even when exhaustive logical evidence is lacking.


## 7. Setup and Implementation Details

Optimized for Apple Silicon (M-series) architectures.

### Base Environment

```sh
# Create and activate the conda environment
conda env create -f environment.yml
conda activate baby_agi

# Pull the base model
ollama pull qwen2.5:7b-instruct-q4_K_M


```

#### ⚠️ Initial Run Requirement (Model Cache)
Before running the project for the first time, you MUST temporarily update `config.py` to allow the vision model (Qwen2-VL) to download from Hugging Face:
* Set `USE_LOCAL_MODEL_CACHE_ONLY = False` (Switch back to `True` once the weights are fully cached locally).

* **Requires Ollama**: [Ollama for MacOS](https://ollama.com)

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

### Demo Logs

You can find a short execution example in the `demo/ko` folder. *(Note: English demo logs are not yet available, but the core mechanics are identical to the Korean demo.)*

* In the demo, the agent initially answers that Dad lives in "Seoul" based on its past memory.
* It quickly accepts the new information ("Dad is currently in London"), and after some trial and error, successfully learns a complex condition: "Greet Dad in Korean, but greet others in English."
* By the end, it strictly adheres to this rule even when prompted in English. You can also observe it initiating conversations spontaneously during idle periods.
* The included runtime debug screenshots show exactly how its emotional state, arousal, valence, mood, RPE, recent chat, and internal monologue update in real-time.

## Project Notes & Attributions

* **AI-Assisted Prototyping**: As recorded in the repository's initial commit history, this project began as a rapid prototype scaffolded with Gemini and ChatGPT. Generative AI was leveraged to quickly conceptualize and iterate ideas, while the final codebase and overall architecture were extensively refactored, tailored, and rewritten manually.
* **Gratitude for Foundational Insights**: This project is deeply inspired by and indebted to the pioneering research established across various fields, including cognitive science, developmental psychology, and control engineering. I also extend my sincere thanks to the brilliant tech communicators whose educational videos made these complex concepts so accessible. While this repository does not follow strict academic citation standards, I have the deepest respect and gratitude for everyone whose foundational work made this implementation possible.
