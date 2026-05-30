# TODO: Future Design Ideas

## Short Term: Self-Regulation and Cognitive Digestion

The core theme for short-term improvements is autonomy: the ability to think, regulate, and push back when necessary. This is not just about adding more features; the focus is on reducing system overhead and building a sustainable architecture.

### Sleep and Memory Logic Improvements

* At the moment, the flashback interval is almost fixed. During sleep, this can keep creating additional short-term memories and limit how much `fatigue` can recover. In practice, after a certain point, sleep can continue indefinitely.
  * The flashback interval should become dynamic based on accumulated short-term memory volume, arousal, fatigue, and sleep pressure. In other words, when there is more memory to digest or arousal is high, dreams should surface more often; once enough memory has been cleaned up and sleep pressure drops, dream density should naturally decrease.
* Memory retrieval should also be adjusted so that, in addition to emotional value, it is affected by the agent's current available-resource state.
  * This should connect with the later `Burden` logic, allowing the system to reflect available resources in its judgment.

### Hardening the Diary Pipeline

* **Data Ingestion Loop:** Move beyond the current write-only diary. The internal monologue loop should read and analyze previous diary entries, then dynamically feed them back into the cognitive pipeline.

### Internal Monologue and Self-Reflection Loop

* **Pre-Action State Check:** When a sharp emotional swing occurs due to an external stimulus, or right before tool execution, add a step that recursively summarizes the situation in one or two sentences to ground the state.
* **Autonomous Reasoning Trigger Tooling:** Explore a dedicated tool interface that lets the system trigger internal monologue and step-by-step reasoning on its own when emotional state, available energy, or time suggests deeper judgment is needed.
  * **Extra Reasoning During Replies:** When an answer is required, encourage additional reasoning while respecting time constraints.
  * **Metacognition:** If large RPE stimuli keep repeating (`RPE momentum` is high), trigger additional evaluation of the current internal/external situation.
    * This is similar to wondering, “What is going on here?” when unexpected events keep happening.
  * **Cognitive Load Control:** If reflection runs too long, increase system fatigue to prevent context pollution and infinite loops. Since forming or cleaning up memory already carries compute cost, that cost can be used here.

### Cognitive Load Evaluation and Autonomous Pushback

* **Limit Awareness and Hallucination Suppression:** When a request is too ambiguous or requires calling a high-failure-rate tool, increase the `Burden` index to prevent the LLM from hallucinating an answer just to satisfy the prompt.
* **Execution Refusal and Resolution Mechanics:** If `Burden` exceeds the threshold, stop tool execution and surface an exception to the upper layer. However, instead of simply giving up, pair this with a resolution loop that uses internal monologue to reassess the load or break the task into smaller pieces before retrying.

### Simulating Extinction Learning for Trauma Dilution

* **Protecting Emotional Topology:** Avoid forcibly deleting specific negative memory nodes from the Vector DB, since doing so could destabilize the entire emotional space.
* **Geometric Dilution:** When a strongly negative node is retrieved, densely place positive-feedback or stable-state nodes near that vector coordinate. This lowers the probability that the negative memory will be retrieved in isolation, mimicking emotional desensitization.
* Also simulate the rare re-digestion and re-storage of existing emotional or long-term memories.
  * The emotion DB should update slowly through new experiences without directly overwriting the existing emotional space in bulk. The long-term memory DB may also occasionally perform merge operations.

### Empirical Weight Adjustment (Aging Algorithm)

* **System Maturity Metric:** Define the cumulative memory count $N$ as the system's maturity indicator.
* **Dynamic Solidification of System Constants:** As $N$ increases, gradually raise momentum weights and decay constants that support early plasticity, creating stability so minor stimuli do not easily disrupt emotional trajectories.
* **Subjectivity and Internal Belief Formation:** As $N$ increases, reduce the attention weight of the system prompt (innate instruction). Instead, place self-accumulated diary data or memory-derived data higher in the context, allowing the agent to gradually prioritize lived experience over static instructions.

### Sandboxed Web Search

* **Network Noise Isolation:** If raw web spam and noise enter while the agent is still highly plastic, the whole embedding space may become polluted.
* **Phased Rollout and Acceptance Evaluation:** Gradually open search capability using mechanisms such as Safe Browsing APIs, while quarantining results inside a volatile working-memory buffer. Only after user approval or internal-monologue validation should they be reflected into long-term memory.
* **Rate-Limit Guardrails:** Apply a strict physical cap on tool-call count per time unit.

### `Daycare` Learning Environment

* **Autonomous Learning from External Materials**: Right now, Baby mostly learns from what Dad explicitly teaches it. A future goal is to let Dad say “look this up” and have Baby study temporary materials such as text snippets, images, web search results, or public LLM API responses.
* **A Learning Space Separate from the Playpen**: If the `Playpen` is the boundary that keeps action safe, the `Daycare` is a supervised learning buffer. New information should be read, summarized, compared, and reflected on before it is allowed to affect long-term memory.
* **Commit Only After Verification**: External materials should not directly rewrite Baby’s personality or long-term memory. They should first pass through source tracking, confidence checks, Dad’s approval when needed, and internal monologue-based review.

### Expanding Visual Perception

* **Isolated Image Feeding:** Beyond periodic host-screen capture, implement a branch where an external image file dropped into a specified path temporarily pauses screen capture, feeds that external image into the vision engine first, then automatically deletes it.

### Refining the Visual Perception Loop

* **Selective Use of the Expensive VLM**: The current vision stack is closer to a periodic full-screen summarizer, which means the expensive VLM is being used in a fairly blunt way. The next step is to decide whether the screen is worth looking at before invoking the model, based on screen-change magnitude, text density, Dad’s current workflow, and Baby’s emotional/arousal state.
* **A Thalamus-Like Sensory Gate**: Instead of passing every visual summary straight into the reasoning loop, the system should filter out low-value changes and only promote perception into working memory when it is relevant to emotion, memory, or the current task.
* **Keep Perception Separate from Action**: The vision module should remain an eye, not a hand. It should not gain direct click or keyboard authority, but important observations can still trigger Baby to speak to Dad or leave a trace in memory.

### Emotional Stabilization and Shock Simulation

* **Desensitization Trigger:** If Reward Prediction Error ($RPE$) exceeds an extreme threshold (for example, $RPE > 0.85$), temporarily set stimulus receptivity to zero to simulate emotional numbing and physically protect the system from collapse.
  * Completely blocking learning during extreme stimuli is useful for system protection, but it may reduce adaptive plasticity. It is currently excluded and will be revisited later alongside the personality drift model.

### Constant-Based Personality Drift

* Improve the system so that the many fixed hyperparameters currently defined in code can change. For example, the current weights between semantic similarity and emotion during memory retrieval are fixed; the goal is to identify what kind of constant each one is and let it change depending on the situation.
  * Some constants are closer to learned personality shaped by the individual agent's experience; some are closer to the individual agent's innate temperament; others are closer to species-level innate temperament.
  * Group these constants into a few categories, then define `personality drift` based on how much each group changes over time.

---

## Mid-to-Long Term: Abstract Emotion and Personality Experiments

In the mid-to-long term, the project explores experimental changes that maximize the agent's individuality and unique emotional expression. Since complex architecture does not automatically guarantee better performance or interpretability, these changes should be approached with careful quantitative validation.

* **High-Dimensional Emotion Space:** Move beyond the simple 3-axis (`JOY / SAD / ANG`) system and design a higher-dimensional emotional matrix capable of representing richer emotional combinations.
* **Geometric RPE Auxiliary Signal:** Rather than immediately replacing the existing 3-axis RPE, experiment with also using the embedding distance between an “expected emotional sentence” and an “actual stimulus sentence” to create a more nuanced surprise signal.
* **Multi-Entity Interaction:** Move beyond the single-user environment and study context separation and multi-relationship handling when different external entities or multi-sided interactions enter the system.
