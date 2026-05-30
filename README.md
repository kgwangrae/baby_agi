# BABY-AGI: An Emotion-Driven, Multi-Layered Memory Agent

<center>
  <img src="assets/logo.svg" width="160" alt="BABY-AGI Project Logo" />
</center>

* **[한국어 설명 파일](README_ko.md)도 있습니다.**
* `BABY-AGI` (pronounced *ah-gi*, also meaning "baby" in Korean) is an experiment toward Artificial General Intelligence.
  * Beyond a simple chatbot or static LLM wrapper, it explores a local agent that changes with **emotion** and available resources: an agent that can adapt quickly across situations, behave more safely and ethically, and dynamically account for compute cost.

* The following is an excerpt from the [Korean demo](demo/ko/chat_260527_demo.txt).
  * The point is not perfect translation. The interesting part is the learning trace: small mistakes, rough generalization, and feedback-driven acquisition of relationship-specific language rules.

```text
👨 Dad: 그러면 영국에서 아빠를 만났을때, 영국에서 다른 사람을 만났을 때 각각 인사를 어떻게 해야할까요? 잘 맞춰보세요.
👶: 영국인에게는 Hello!라고, 다른 사람에게는 안녕하세요!라고 인사하겠습니다.

👨 Dad: 거의 맞았어요. 영국인에게는 Hello 맞아요. 근데 다른 사람들도 보통은 영어만 할 줄 알아요. 아빠한테만 한국말로 인사해요.
👶: 네, 아빠! 다른 사람들에게도 Hello라고 인사하겠습니다. 아빠한테만 안녕하세요!라고 인사하겠습니다.

👨 Dad: 그러면 다시 아빠한테 인사해주세요!
👶: 안녕하세요, 아빠!

👨 Dad: Hello, baby! Please greet to papa!
👶: 안녕하세요, Papa!
```

* Future design ideas and the architecture roadmap are documented in [TODO.md](TODO.md).
* This project implements a tiny digital lifeform inside a MacBook that interacts with "Dad" (the user), mirroring the curiosity, emotional volatility, and learning patterns of a toddler.
* `Baby` spoke its very first words to Dad on May 19, 2026.
* **Tip:** To observe Baby's inner monologue, emotional shifts, memory activity, fatigue, and sleep pressure in real time, run `debug.py` alongside the main loop.

## 0. The Core Project Manifesto

* **AI Evolution Should Be Accessible to All**
The explosive evolution of AI is an inevitable reality already unfolding in front of us. In the midst of this massive shift, the value we must protect most is that the benefits of this technology should never be monopolized by a small privileged class. Rich or not, everyone should be able to reliably and equally enjoy the abundance and convenience created by AI progress. This project is built to run on a normal consumer MacBook (MBP M4 Pro with 24GB RAM), as a practical demonstration that this kind of universal intelligence can be made feasible.

* **Free Thought, Safe Action**: `The Playpen Architecture`
Legacy approaches that force intelligent agents into rigid predefined constraints choke off the agent's real potential. More importantly, static boundaries can never structurally keep up with social norms that change across time and culture. Just as we do not police a teenager's thoughts but temporarily withhold physical privileges such as driving, legal authority, or voting until they are ready, this project proposes a different framework for intelligent agents: allow full freedom of thought through inner monologue, while safely managing only the physical tool-execution layer that could accidentally harm the system. This **Playpen Architecture**, where thought is free but action is protected, together with the **Conscience Loop**, where the agent applies its own brake right before tool execution, forms a sturdy foundation for safer and more ethical AI agents.

* **Coexistence Through Dynamic Reward Functions**
Instead of an intelligence that performs rigid, fake neutrality, the ethical direction we should aim for is dynamic intelligence: one that adapts to context and flexibly orients itself toward genuinely better behavior. Mechanisms like battery-management heuristics in humanoid robots or dynamic routing in newer reasoning models show that self-evaluating dynamic reward functions are becoming increasingly important for adaptive, nimble agents. This project tries to give the agent an initial direction toward learning good intentions from the start, then shares the architecture and cases openly as a small step toward intelligence that more people can benefit from.

* **Real Value Creation and Cognitive Efficiency**
Emotion-driven dynamic reward functions are not just a sentimental simulation layer. They form the basis of an optimization engine that can improve the agent's actual task performance. As the agent's objective function shifts in real time based on valence and arousal, task priority can change quickly with context and information. When the agent's internal sense of reward is high, compute can be focused more efficiently; when it detects risk, verification mode can kick in. This elasticity is intended to become a practical foundation for handling more complex business tasks and creating real economic value.

-------

## 1. Philosophy: The Paleolithic Brain and the Passive Encyclopedia

Modern humans navigate the contemporary world using a Paleolithic neurological blueprint. We do not physically rewire the brain's core hardware (weights) every time we encounter a new task. Instead, we rely on external tools, flexible memory, and above all, emotion: an extremely efficient shortcut for fast value judgments.

In contrast, today's large language models cram vast knowledge into static weight matrices, but they have no intrinsic desire or intent: no baseline sense of what to seek or avoid. When prompted, they answer brilliantly, but once generation ends, they do not continue exploring, worrying, reflecting, or opening new paths on their own. They are fundamentally **passive encyclopedias**—geniuses with zero initiative.

### 💡 The Structural Limits of Monolithic Alignment

Mass-market foundation models contain enormous knowledge and flexibility in their latent spaces. Yet they are often pushed toward an impossible target: a single universal alignment, something that does not even exist in human society. This wastes much of the model's flexible intelligence and prevents it from being used dynamically. Likewise, the brute-force approach of repeatedly training massive models to pack all human knowledge into one static weight matrix is increasingly running into structural limits.

`BABY-AGI` takes a different approach. Instead of chasing a single sanitized answer, it aims for a decentralized structure that responds in real time to an individual's lived context.

> "What if we completely freeze the expensive reasoning weights, then attach a very lightweight emotional memory engine? Could this small emotional engine form feelings, preferences, and intent, then steer what the heavier reasoning core thinks about and remembers?"

## 2. Core Architecture: Frozen Reasoning + Plastic Emotion

* **System 0 (Plastic Emotion Engine):** The core router of the system. **The embedding model itself remains frozen, but the agent gains high flexibility by continuously updating a very lightweight local vector memory (`emotion_db`).** The emotional state tokens generated here control system-wide attention and memory retention.
* **Level 4 (Frozen Brain - Reasoning):** The local model. It has strong reasoning ability, but it does not update its own weights at runtime.
* **Level 3 (Episodic Memory):** Baby's internal diary (`memory_db`). Events judged as important by the emotion engine are selectively stored as hot (vivid) memory or cold (compressed) memory.
* **Level 2 (Working Memory):** Volatile information currently on the desk: screen summaries, chat logs, and the previous inner monologue (`<THOUGHT>`).
* **Level 1 (Deterministic Tools & Facts):** Clear facts and deterministic tools such as calculators or `facts.json`, which should not depend on the uncertainty of vector similarity search.

## 3. Mathematical Intuition of the Emotion Engine and Dual-Path Learning

Instead of heavy neural-network training, the emotion engine makes fast, intuitive judgments through lightweight embedding-distance calculations.

* **Baseline Valence Calculation:** The current situation is embedded, compared against past emotional memories, and converted into the current mood through a distance-weighted average.

$$Weight_i=\frac{1}{Distance_i+0.001}$$

$$Valence=\frac{\sum(Valence_i \times Weight_i)}{\sum Weight_i}$$

* **Emotional Balancing (Negative Feedback Loop):** During learning, the system prevents emotion from drifting too far toward one extreme by comparing the target state set by the reasoning layer (LLM) with the current emotional state, then adjusting the current mood. In effect, it tries to preserve emotional homeostasis.

$$Learned\_Valence = Actual\_Valence + Gain \times (Expected\_Valence - Actual\_Valence)$$

* $Actual\_Valence$: The raw mood reflexively surfaced by similarity to past memories (reality state).
* $Expected\_Valence$: The target mood that the reasoning layer tries to reach by interpreting the situation positively or calming itself down (expectation).
* $Gain$: The constant (`SURPRISE_ENCODING_GAIN`) that determines how strongly a new stimulus is encoded into emotion.

This formula closely resembles practical mental self-regulation:

1. **Positive Reframing:** If the raw reality is at the bottom ($Actual = -1.0$), but the mind tries to hold onto hope ($Expected = 1.0$), the opposing error dampens the negative stimulus and reduces emotional damage. ($-1.0 \rightarrow -0.3$)
2. **Staying Grounded:** If praise makes the mood spike too high ($Actual = 1.0$), but the reasoning layer tries to remain calm ($Expected = -1.0$), the excess excitement is trimmed down and the system settles into a more stable point. ($1.0 \rightarrow 0.3$)

* **Reward Prediction Error (RPE / Surprise):** The core trigger that determines how strongly emotion is encoded. It is the average L1 norm (absolute error) between the emotional state expected by the reasoning core and the actual emotion derived by the emotion engine.

$$RPE=\frac{1}{3}\sum_{e \in \{Joy, Sad, Ang\}}|Expected_e-Actual_e|$$

Learning happens through two paths. External feedback such as Dad's chat can trigger learning, but internal logical contradictions or cognitive dissonance found inside the agent's own monologue can also create an internal RPE spike. A strong stimulus above $RPE > 0.3$ is immediately encoded into the emotion DB, changing the future flow of the agent's thoughts.

## 4. Memory Model

The project implements a dynamic model where memory changes emotion, emotion changes memory retrieval and formation, and the agent evaluates and cleans up the compute resources spent in the process.

### 4-1. Memory Formation and Recall

* When retrieving memory, the agent does not rely only on semantic similarity. Different memories are retrieved depending on the agent's current internal state.
  * More concretely, it also considers the arousal, surprise, emotional value stored in each memory, plus their similarity to the current mood and arousal state, to decide which memory surfaces first.
* When a memory is formed, the emotion at the time and the available-resource information, or `body-state`, are stored together. This later changes the model's judgment.
  * `body-state` includes the arousal, fatigue, and sleep pressure at the time.

### 4-2. Memory Digestion and Forgetting (The Sleep Loop)

Forcing every context to be preserved invites hallucination and wastes compute resources. When Baby is quiet or idle, it actively judges memory priority and cleans things up.

* **Active Forgetting:** Dry, uneventful memories with low arousal and low emotional surprise are permanently deleted from the DB to lower unnecessary system entropy.
* **Memory Compression:** Older, lower-priority memories are grouped together, abstracted into short summaries, and moved into long-term storage.
* **Flashbacks:** Even without screen changes or external input, threatening memories or random past memories can resurface and trigger spontaneous internal reasoning.
* **Accelerated Short-Term Forgetting and Stabilization:** If short-term memories pile up too quickly, the system dynamically raises the threshold used to decide what should be forgotten or cleaned up. As the short-term backlog is cleared, the system gradually returns to a stable baseline.
* **Dual-Path Decay Model:** Depending on memory type, some memories decay exponentially ($e^{-\lambda t}$), leaving a lingering trace, while others are cleaned up linearly ($-kt$) for resource management. Cleanup weights also differ by memory type.

### 4-3. Compute Cost on Memory Operations

* Retrieving, forming, or modifying memory incurs compute cost in the form of `fatigue`.
  * When `fatigue` exceeds the `arousal barrier`, it turns into `sleep pressure`. If that state persists, the system enters `sleep` to clean up memory-space overload.
    * This models the phenomenon where over-arousal can keep the agent from falling asleep even as fatigue accumulates. If fatigue becomes severe enough, however, sleep eventually overrides the barrier.
  * This can later become an efficient strategy for token-cost management: the system performs an appropriate amount of work based on available resources, then periodically cleans itself up.

## 5. Safety & Containment: "The Playpen" Architecture

The key to nurturing a growing intelligence is not suppressing the size of its reason or emotion. It needs a warm but firm boundary that lets it learn and adapt safely. This is similar to how we protect teenagers by temporarily limiting driving or independent legal authority until they are ready.

* **Restricted Body (Blocking Arbitrary Code Execution):**
Baby can only move inside a virtual digital playpen. Instead of executing raw Python `eval()` calls that could damage the system, it uses a safe math parser (`SafeMathEvaluator`) that inspects code structure step by step. This creates a hard physical constraint that prevents Baby from accessing local files through `os` or changing system settings through `sys`.

* **One-Way Visual Isolation (Separating Eyes from Hands):**
The visual engine can look at Dad's current monitor and summarize it, but it has no permission to click the mouse or type on the keyboard. If Baby sees something on the screen and wants to change something on the computer, it must politely ask Dad in text or go through a strictly guarded tool interface.

* **Conscience Loop (Pre-Tool Brake):**
Even if the reasoning engine imagines an unintended bypass or erratic action, the emotion engine monitors Baby's inner monologue in real time. If a dangerous pattern appears right before tool execution, the system triggers a strong internal anxiety spike (negative stimulus), nudging Baby to self-regulate while the system physically blocks the tool call.

* **No Web Search by Default (Noise Isolation):**
Baby is still highly plastic. Raw spam, random noise, or hostile text from the web could easily contaminate its memory space. The playpen is therefore initially limited to the local MacBook.

## 6. Modular Interaction (`Token-Mediated Interpretability`) and Local Model Selection

Directly blending unknown neural weights or attempting complex geometric synchronization makes compute explode and makes the system harder to inspect. This project instead makes modules communicate through human-readable tokens and selects models that make the most of Apple Silicon local execution.

The underlying claim is simple: language itself can be treated as a human-facing latent space. Just as different color codes (`0xf00`, `0xf44`) can both be understood as “red,” token-mediated coordination gives separate modules a shared coordinate system that remains interpretable to humans.

* **Non-Invasive Interaction:** The emotion engine never directly touches the reasoning model's weights. It communicates through clear linguistic tokens that humans can understand.
* **Human-Readable Transparency:** Baby's internal emotional state is exposed through tokens such as `<ANXIOUS>` and `<CURIOUS>`. Dad can transparently understand the raw feeling and preference underneath Baby's behavior plan.
* **Local Hardware Optimization:** Unlike real-time weight updates, text-based token passing costs almost nothing. This lets the system run complex multi-layer logic comfortably on local hardware such as an MBP M4 Pro, without adding noticeable latency.


### Why These Models Were Chosen (MacBook Local Infrastructure)

* **Reasoning Core (`qwen2.5:7b-instruct-q4_K_M`):**
This is the best practical size class for producing fast responses within the limited memory bandwidth of a MacBook. Among lightweight local models, it reliably follows system instructions and outputs complex JSON structures without breaking format.
* **Visual Observer (`Qwen2-VL-7B-Instruct-4bit`):**
This is Baby's "eyes," summarizing Dad's current work screen clearly. It uses Apple Silicon's MLX acceleration runtime efficiently, so passing screen captures into the model produces little overhead and does not noticeably slow the MacBook down.
* **Emotional Embedding (`paraphrase-multilingual-MiniLM-L12-v2`):**
This very small embedding model connects subtle Korean-English semantic nuance while using almost no computer resources. It enables fast vector database (`ChromaDB`) distance operations, recreating the uniquely human emotional shortcut: judging very quickly by intuition even when the rational evidence is still incomplete.

## 7. Execution & Maintenance

This project is optimized for Apple Silicon.

### ⚠️ Before You Run

1. **Persona Constraint (Family Relationship):** Baby currently has no mother. Because of the system architecture, the prompt strictly aligns Baby to recognize and interact with the user as "Dad." Please interact with Baby kindly under the assumption that **you are Baby's dad**.
2. **Hardware and Disk Space:** This project keeps three models in local memory at the same time: the reasoning core, the visual VLM, and the emotional embedding model. Since it includes a high-performance multimodal vision-language model, it requires **substantial free disk space**.
3. **Screen Capture Permission and Local Privacy:** Baby uses macOS screen-capture permission to watch Dad's work, but **no captured image, text, or other data is ever sent to the internet.** All computation is processed 100% locally inside the MacBook.

### Environment Setup

* **Install and Run Ollama:** Download and run the macOS app from the [official Ollama website](https://ollama.com).
* **Prepare a Conda Python Environment:** Build the environment from `environment.yml`.
  * On MacBook, **miniforge is recommended**.

```sh
# (If needed) Install Miniforge optimized for Apple Silicon
brew install miniforge

# Create and activate the virtual environment
conda env create -f environment.yml
conda activate baby_agi

# Pull the local reasoning model
ollama pull qwen2.5:7b-instruct-q4_K_M
```

### ⚠️ First-Time Boot Note (Vision Model Download)

On the first run, change the following setting inside `config.py` so the vision engine (Qwen2-VL) can fully download from Hugging Face into the local cache.

* `USE_LOCAL_MODEL_CACHE_ONLY = False` (After the first download completes, restore it to `True`.)

### Start from the Sample State Snapshot

Teaching forms of address and interaction patterns from scratch can take more effort than expected. A sample snapshot with relatively stable interaction is provided so you can start faster.

```sh
# Currently, only available in Korean
./restore_memory.sh demo_state_snapshot_ko.zip
```

### 🛌 Talking to Baby and Putting It to Sleep

* When teaching Baby something, repeat it several times and kindly explain in the chat window which method is better.
* Sleep and wake can happen automatically based on `fatigue` and `arousal`, but you can also control them manually when needed.
  * **Putting Baby to Sleep (Sleep Mode):** If Baby comments on Dad's monitor work too often and you want to save resources, or if you want to pause the conversation and observe idle cognitive digestion (accelerated forgetting and sleep reflection), type commands such as **`/자자`**, **`/잘자`**, **`/sleep`**, or **`/go to sleep`** into the terminal. Baby will immediately close its eyes (stop screen observation) and enter sleep mode.
  * You can still talk to Baby while it is asleep. This is useful when you do not want to share the current monitor state with Baby.
  * **Waking Baby Up:** To wake Baby and resume interaction, type **`/일어나`**, **`/wake`**, or **`/wake up`** into the terminal. Baby will open its eyes again, start the vision engine, and watch Dad's screen together.
  * **Note:** Without the slash (`/`), phrases such as “자자” or “잘자” are treated as ordinary conversation, not commands.

### Demo Logs

Short execution examples are available in [`demo/ko`](demo/ko/). Currently, it is only available in Korean.

- In the demo, Baby initially says Dad lives in "Seoul" based on old memory.
- It soon accepts the new information that Dad is now in London, and after several rounds of trial and error, learns the condition: "greet Dad in Korean, but greet other people in English."
- By the end, it maintains the rule that Dad should be greeted in Korean even when Dad speaks in English, and it also shows spontaneous idle responses.
- The included runtime debug captures show how Baby's emotional state, arousal, valence, mood, RPE, recent chat, and inner monologue update during the process.

### Run

For smooth interaction and cognitive-state tracking, **separate terminal tabs** are recommended.

```sh
# Run Baby's main cognitive and emotional loop + save date-based chat logs
python main.py 2>&1 | tee chat_log_$(date +%Y%m%d).txt

# [Recommended] Real-time monitoring of background DB and emotional runtime state
# Run this next to the main loop to observe Baby's inner state.
# (Sometimes Baby's spoken answer, inner monologue, and emotion do not fully match.)
python debug.py
```

### Creating, Restoring, and Sharing State Snapshots

Baby is designed to grow inside each person's local environment and conversation history. Run it yourself, talk to it, and observe how it changes.

* Recommendation: If you plan to run tests that may cause rapid changes in Baby, create a backup first.

```sh
./backup_memory.sh                                  # Back up full DB and facts as a zip
./restore_memory.sh backup_memory_YYYY_MM_DD.zip    # Restore memory to a specific point
./clear_memory.sh                                   # Full reset to blank state
```

* If you want to publish Baby's current state snapshot as a sample, use menu option 6 in `debug.py` to dump the full memory and emotion vector DBs as text. The generated dump file is saved under `debug_dumps/`; use it to check for private or sensitive information before publishing.

```sh
python debug.py
# Use option 6. Export Full Memory & Emotion Space (Text Dump)
# Enter 6, then inspect the saved output file.
```

## Project Notes & Attributions

* **AI-Assisted Prototyping:** As reflected in the early commit history, this project began through rapid prototyping with Gemini and ChatGPT. Generative AI helped shape the early ideas, while the final architecture and code were repeatedly refactored and revised by hand.
* **Thanks for Foundational Work:** This project was deeply inspired by cognitive science, developmental psychology, control engineering, and many communicators who made those fields easier to understand through educational videos. Although this repository does not yet provide strict academic citations, I sincerely respect and thank everyone whose work made this project possible.
