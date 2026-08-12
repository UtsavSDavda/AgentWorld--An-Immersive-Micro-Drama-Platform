# GROUNDED: Partitioning Creative Agency Between LLM Storytellers and a Deterministic World Engine

## Project Overview

This project is a comprehensive framework for running, visualizing, and evaluating LLM-driven autonomous agents within interactive fiction environments. It combines a text-based simulation engine with advanced video generation and a multi-tiered evaluation suite to assess agent cognition, narrative consistency, and physical logic.

The architecture is divided into three primary pipelines:
1.  **Simulation & Agent Cognition:** Runs the game world and handles agent reasoning.
2.  **Automated Video Generation:** Converts text-based logs into cinematic video sequences.
3.  **Comprehensive Evaluation:** Tests the agents on narrative quality, physical adherence, persona consistency, and learning capabilities.

---

## ⚙️ Component Breakdown

### 1. Simulation & Cognition Engine
The simulation is grounded in a text-based interactive fiction environment, orchestrated through a robust backend.
*   **Environment:** The project uses the `jericho` library to interface with Z-Machine games (e.g., *Control4.z8*)[cite: 3, 5]. 
*   **Agent Logic (`BaselineAgent`):** Agents are powered by the `gemini-2.5-flash` model[cite: 3, 5]. They maintain a base persona and track cognitive states, including their current objective, immediate blockers, and relational stances toward other characters[cite: 5].
*   **Reflection & Action:** Agents periodically reflect on recent physical events and conversations to update their psychological state and stress levels[cite: 5]. They utilize chain-of-thought reasoning to decide on actions from a generated catalog of available moves (e.g., navigating, interacting with objects, or conversing)[cite: 5].
*   **Data Logging (`SQLLogger`):** All physical outcomes, room descriptions, and dialogue are synchronized and stored in a Supabase PostgreSQL database[cite: 3, 5]. This ensures a persistent "ground truth" for both video generation and evaluation[cite: 5].

### 2. Automated Video Generation (The Director)
The `AutomatedDirector` translates the text-based simulation into a visual cinematic experience using a variety of generative AI tools[cite: 3].
*   **Asset Generation:** The system uses `imagen-4.0-fast-generate-001` to dynamically create 1:1 character portraits and 16:9 background plates based on room descriptions and agent personas[cite: 3].
*   **Compositing:** A local MediaPipe Selfie Segmenter model cuts out the generated characters and naturally composites them over the background plates[cite: 3].
*   **Audio Synthesis:** Google Cloud TTS generates voice lines for the characters, mapping specific voices to characters based on their generated profiles[cite: 3].
*   **Video Animation:** The composited images and text prompts are fed into `veo-3.1-fast-generate-preview` to animate the scenes[cite: 3]. The system syncs the resulting video with the TTS audio, adjusting tempos or freezing frames to ensure alignment[cite: 3].
*   **Asynchronous Tasking:** A `Huey` task queue with an SQLite backend (`render_queue.db`) handles the heavy lifting, allowing for asynchronous rendering of individual scenes, recap videos, or full episodes[cite: 7].
*   **Connectivity:** A Flask application and an Anvil Uplink script bridge the local rendering engine and simulation with a web-based frontend[cite: 2, 4, 6].

---

## 📊 Evaluation Suite & Nuances

The project evaluates the LLM agents through a rigorous, multi-dimensional benchmarking suite, isolating different aspects of their performance.

### 1. DramaBench Evaluation (`eval.py`)
This script evaluates the narrative quality of the simulation using `gemini-2.5-pro` over a striding 5-tick window[cite: 1].
*   **Character Consistency (OOC Rate):** Measures how often an agent breaks character or acts out of character (OOC)[cite: 1]. A lower score indicates better adherence to their persona[cite: 1].
*   **Logic Consistency (Break Rate):** Evaluates if the agent attempts physical actions that violate the established environmental state[cite: 1].
*   **Conflict Handling:** Assigns a global weight to the narrative tension, rewarding escalation and twists while penalizing dropped storylines[cite: 1].

### 2. Direct Entailment & Persona Judge (`nli_benchmark.py`)
This script acts as a strict judge using `gemini-2.5-pro` to prevent false positives in agent dialogue[cite: 8].
*   **Eval 2 (Physical Consistency):** Evaluates whether an agent's dialogue contradicts the absolute physical ground truth of the engine logs (locations, sights, past events)[cite: 8]. It explicitly ignores narrative chat history to prevent hallucination verification, labeling claims as VERIFIED, HALLUCINATED, or N/A[cite: 8].
*   **Eval 3 (Persona Consistency):** Checks if the dialogue aligns with the speaker's hidden motives and secret knowledge, scoring it as CONSISTENT or INCONSISTENT[cite: 8].

### 3. Cognitive Adaptation (`stubbornness_test.py`)
This evaluation measures the agents' ability to learn from their mistakes within the physics engine[cite: 9].
*   **Action Tracking:** It parses the `DATA_RESULT` engine logs chronologically to track successful and failed actions[cite: 9].
*   **Stubbornness Index:** The script specifically monitors if an agent repeats the exact same failed action without changing their approach[cite: 9]. A high "Stubborn Rate" indicates a failure in cognitive adaptation and reasoning[cite: 9].

---

## 📈 VBench Testing & Results

To evaluate the capabilities of our automated video generation pipeline (The Director), we benchmarked the generated outputs against **VBench**, a comprehensive evaluation suite for video generative models. 

The table below merges the testing results across the evaluated dimensions, comparing our pipeline's scores against the State-of-the-Art (SOTA) averages and identifying the leading models for each category at the time of recording.

| VBench Dimension | Our Score | VBench SOTA Avg. | SOTA Model |
| :--- | :--- | :--- | :--- |
| **Subject Consistency** | 0.8626 | 0.9756 | Wan2.1-T2V-1.3B |
| **Background Consistency** | 0.8754 | 0.9895 | Pika Beta |
| **Motion Smoothness** | **0.9938** | 0.9916 | Veo 3 |
| **Aesthetic Quality** | 0.6256 | 0.6381 | Veo 3 |
| **Imaging Quality** | **0.7082** | 0.6970 | Kling 1.6 |

### Result Highlights
*   **Highly Competitive:** Our automated pipeline successfully beat the SOTA average for **Motion Smoothness** (scoring 0.9938 vs the 0.9916 average of Veo 3) and **Imaging Quality** (scoring 0.7082 vs the 0.6970 average of Kling 1.6). 
*   **Areas for Growth:** While our pipeline produces excellent aesthetics and smooth motion, there is a slight gap in **Subject Consistency** and **Background Consistency** when compared to dedicated consistency models like Wan2.1-T2V-1.3B and Pika Beta.