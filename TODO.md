# TODO: Future Architecture & Design

## 1. Version 2 (V2): Self-Regulation & Cognitive Assimilation

The core theme for V2 is **autonomy: the ability to think, regulate, and push back when necessary.** This is not just about bolting on new features; the focus is on reducing system overhead and building a sustainable, long-term architecture.

### Hardening the Diary Pipeline

* **Data Ingestion Loop**: Moving beyond a write-only diary, the internal monologue loop will now read and analyze past diary entries, dynamically feeding them back into the cognitive pipeline.

### Internal Monologue & Self-Reflection Loop

* **Pre-Execution Sanity Check**: Before executing tools or after a sharp emotional spike, the agent will recursively summarize the situation in 1-2 sentences to ground its state.
* **Cognitive Load Throttling**: If the reflection process drags on, the system will artificially bump up a 'fatigue' metric to proactively break infinite loops and prevent context pollution.
* **Autonomous CoT (Chain-of-Thought) Trigger**: A dedicated tool interface is under exploration to let the agent spontaneously trigger internal monologues to perform step-by-step reasoning based on its emotional state, available energy, or time.

### Cognitive Load Evaluation & Autonomous Pushback

* **Acknowledging Limits to Curb Hallucinations**: When hit with ambiguous prompts or high-failure-rate tool requests, the agent will bump up its `Burden` index. This stops the LLM from hallucinating answers just to fulfill the prompt.
* **Execution Pushback & Resolution Mechanics**: If the `Burden` threshold is breached, the agent halts tool execution and throws an exception to the upper layer. However, instead of just giving up, it triggers a resolution loop—using internal monologue to reassess the load or break the task down into smaller chunks for retry.

### Simulating Extinction Learning for Trauma Dilution

* **Protecting Emotional Topology**: Brute-force deletion of negative memory nodes in the Vector DB is avoided, as doing so can shatter the balance of the entire emotional space.
* **Geometric Dilution**: When a highly negative node is retrieved, positive feedback or stable-state nodes are densely packed around its vector coordinates. This mathematically dilutes the probability of retrieving the negative memory in isolation, mimicking psychological desensitization.

### Information Interference & Mean Reversion of Thresholds

* **Accelerated Short-Term Forgetting**: If token I/O and emotional volatility spike, the `FORGET_THRESHOLD` is dynamically raised to protect the short-term buffer. This mimics the biological interference mechanism where cramming too much data overwrites recent memories.
* **Dual-Path Decay Model**: Emotional data decays gradually via an exponential function ($e^{-\lambda t}$) to leave lingering traces, while episodic memory is aggressively pruned via a linear function ($-kt$) to manage system resources—mirroring cognitive forgetting curves.
* **Baseline Reversion**: When the system stabilizes without external stimuli, memory thresholds gracefully revert to their defaults.

### Empirical Weight Adjustment (Aging Algorithm)

* **System Maturity Metric**: The total number of cumulative memory nodes ($N$) serves as the agent's maturity index.
* **Dynamic Solidification of Constants**: As $N$ grows, momentum weights and decay constants (which handle early plasticity) will asymptotically converge upwards. This creates a stabilizing effect, preventing minor stimuli from easily derailing emotional trajectories.
* **Subjectivity & Internal Belief Formation**: As $N$ increases, the attention weight of the system prompt (innate instinct) scales down. In its place, self-accumulated diary entries and the belief DB (`Belief_DB`) are prioritized at the top of the context. This shifts the agent to trust its lived experiences over static instructions.

### Sandboxed Web Search

* **Network Noise Isolation**: Giving an early-stage, highly plastic agent raw web access risks polluting its entire embedding space with spam and noise.
* **Phased Rollout & Validation**: Web access will be gradually unlocked via Safe Browsing APIs. Search results are quarantined in a volatile working memory buffer. They are only committed to long-term memory after user approval or if the agent validates the data via its internal monologue.
* **Rate Limiting Guardrails**: Strict physical caps will be enforced on tool invocation rates per time window to prevent API abuse and sensory overload.

### Expanding Visual Perception

* **Isolated Image Feeding**: Beyond periodic screen captures, a branching logic is being implemented: if an external image file is dropped into a specific path, the agent pauses screen grabbing, ingests the external image through the vision engine, and then auto-deletes the file.

### Emotional Stabilization & Shock Simulation

* **Desensitization Threshold**: If the Reward Prediction Error ($RPE$) breaches an extreme threshold (e.g., $RPE > 0.85$), the system simulates a defense mechanism by temporarily dropping stimulus receptivity to zero, physically shielding the system from a breakdown.
* *Note: Completely freezing learning during extreme stimuli is great for system protection but hurts adaptability. It is currently sidelined and will be revisited alongside the Personality Drift model in V3.*



---

## 2. Version 3 (V3): Abstract Emotions & Personality Experiments

V3 tackles experimental shifts to maximize the agent's individuality and unique emotional expression. Since complex architectures don't automatically guarantee better performance or interpretability, this roadmap will be approached with strict quantitative validation.

* **High-Dimensional Emotion Space**: Moving beyond the basic 3-axis (`JOY / SAD / ANG`) system to a complex emotional matrix capable of representing nuanced, blended states.
* **Pure Geometric RPE Mechanism**: To bypass the limitations of having the LLM predict numerical emotional values, the 'expected emotional context' and the 'actual stimulus' will be embedded. The cosine distance between these two vectors will map directly to the Reward Prediction Error ($RPE$).
* **Complex-Plane Personality Drift**: Personality constants will be modeled not as simple real numbers, but as complex numbers: `[Inherent Trait Magnitude (Real) + Relational Phase/Intimacy (Imaginary)]`. When stimuli hit, the emotional matrix will rotate geometrically on the complex plane, yielding context-dependent inferences.
* **Multi-Entity Interaction**: Expanding beyond a single-user environment to handle context isolation and multi-relational dynamics when interacting with various external entities.
* **Generalized System Prompt Refactoring**: Moving away from hardcoding weight by repetitively copy-pasting instructions in the context. A more robust, generalized prompt architecture will be built to inherently command strong control through its structure alone.