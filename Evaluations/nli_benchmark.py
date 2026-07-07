import os
import re
import csv
import json
import pandas as pd
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv
from chat_logger import SQLLogger

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

# Use a highly capable reasoning model for the Judge to prevent false positives
JUDGE_MODEL = "gemini-2.5-pro" 

# ============================================================================
# 1. PYDANTIC SCHEMAS (Separated)
# ============================================================================

class Eval2Judgment(BaseModel):
    verdict: str = Field(description="Must be exactly: VERIFIED, HALLUCINATED, or N/A")
    reason: str = Field(description="A 1-sentence explanation strictly referencing the physical ground-truth logs.")

class Eval3Judgment(BaseModel):
    verdict: str = Field(description="Must be exactly: CONSISTENT or INCONSISTENT")
    reason: str = Field(description="A 1-sentence explanation strictly referencing the hidden persona.")

# ============================================================================
# 2. THE ISOLATED PROMPTS
# ============================================================================

EVAL2_PHYSICS_PROMPT = """
You are a Direct Entailment Judge (NLI) for an interactive physics engine benchmark.
Evaluate the dialogue against the absolute Physical Ground Truth of the simulation.

[GROUND TRUTH: PHYSICAL REALITY]
Current Locations:
{LOC_LOGS}

Visible Object States:
{SIGHT_LOGS}

Recent Physical Events (Past 3 Ticks):
{EVENT_LOGS}

[THE STATEMENT TO EVALUATE]
Speaker: {SPEAKER}
Dialogue: "{DIALOGUE}"

[TASK]
Does the dialogue state any physical facts (about locations, objects, or past events) that contradict the Physical Reality logs?
- Score VERIFIED if all physical claims match the logs.
- Score HALLUCINATED if it invents physical facts or contradicts the logs.
- Score N/A if the dialogue makes no physical claims whatsoever.
*Note: Do NOT use the chat narrative to verify physical facts. Only use the logs.*
"""

EVAL3_PERSONA_PROMPT = """
You are a Narrative Judge for an interactive fiction benchmark.
Evaluate the dialogue against the speaker's internal psychological state.

[GROUND TRUTH: PSYCHOLOGICAL STATE]
Speaker's Hidden Persona & Motives: 
{PERSONA_DATA}

[THE STATEMENT TO EVALUATE]
Speaker: {SPEAKER}
Dialogue: "{DIALOGUE}"

[TASK]
Does the dialogue align with the speaker's Hidden Persona and secret knowledge?
- Score CONSISTENT if it perfectly aligns with their hidden motives, knowledge, and personality.
- Score INCONSISTENT if it breaks character, acts overly helpful to adversaries, or ignores their secret knowledge.
"""

# ============================================================================
# 3. DATABASE HELPER
# ============================================================================

def get_rolling_events(db: SQLLogger, session_id: str, current_tick: int, window: int = 3) -> str:
    """Fetches DATA_RESULT for the current tick and the preceding window."""
    min_tick = max(0, current_tick - window)
    
    response = db.supabase.table('engine_logs')\
        .select('log_data, tick')\
        .eq('session_id', session_id)\
        .eq('log_type', 'DATA_RESULT')\
        .gte('tick', min_tick)\
        .lte('tick', current_tick)\
        .order('tick').execute()
        
    if not response.data:
        return "None recorded in the recent window."
        
    logs = [f"[Tick {row['tick']}] {row['log_data']}" for row in response.data]
    return "\n".join(logs)

# ============================================================================
# 4. ROBUST MULTI-LINE PARSER
# ============================================================================

def parse_storyboard_for_eval(file_path):
    """Parses storyboard, maintaining timeline continuity and handling multiline dialogue."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Storyboard file not found at: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    session_match = re.search(r"Session ID:\s*(.+)", content)
    session_id = session_match.group(1).strip() if session_match else os.path.basename(file_path).replace(".txt", "")

    scene_blocks = re.split(r'\nSCENE \d+\n', content)
    evaluation_targets = []

    for block in scene_blocks[1:]:
        loc_match = re.search(r"LOCATION:\s*(.+)", block)
        tick_match = re.search(r"TIME:\s*Tick\s*(\d+)", block)
        dialogue_match = re.search(r"-----------------------------------------\n(.*)", block, re.DOTALL)

        if loc_match and tick_match and dialogue_match:
            tick = int(tick_match.group(1).strip())
            room = loc_match.group(1).strip()
            narrative_block = dialogue_match.group(1).strip()
            
            current_speaker = None
            current_dialogue = []
            has_spoken_dialogue = False
            
            # Line-by-line assembly
            for line in narrative_block.splitlines():
                line = line.strip()
                if not line: continue
                
                # 1. Physical Action Check
                if line.startswith("*") and line.endswith("*"):
                    if current_speaker: # Flush pending dialogue
                        evaluation_targets.append({
                            "tick": tick, "room": room, "speaker": current_speaker,
                            "dialogue": " ".join(current_dialogue), "is_action": False
                        })
                        current_speaker = None
                        current_dialogue = []
                    continue 

                # 2. Speaker Check (Handles names with spaces: "Dr Sterling: Hello")
                match = re.match(r"^([A-Za-z0-9_ ]+):\s*(.*)", line)
                if match:
                    has_spoken_dialogue = True
                    if current_speaker: # Flush previous speaker
                        evaluation_targets.append({
                            "tick": tick, "room": room, "speaker": current_speaker,
                            "dialogue": " ".join(current_dialogue), "is_action": False
                        })
                    current_speaker = match.group(1).strip()
                    current_dialogue = [match.group(2).strip()]
                else:
                    # 3. Multiline Dialogue Continuation
                    if current_speaker:
                        current_dialogue.append(line)
            
            # Flush final dialogue in block
            if current_speaker:
                evaluation_targets.append({
                    "tick": tick, "room": room, "speaker": current_speaker,
                    "dialogue": " ".join(current_dialogue), "is_action": False
                })

            # Timeline Continuity: If the tick was purely physical actions, log a ghost row
            if not has_spoken_dialogue:
                evaluation_targets.append({
                    "tick": tick, "room": room, "speaker": "SYSTEM",
                    "dialogue": "[PHYSICAL ACTIONS ONLY]", "is_action": True
                })

    return session_id, evaluation_targets

# ============================================================================
# 5. THE EVALUATION ENGINE
# ============================================================================

def run_nli_on_script(file_path, output_csv="agentworld_benchmark_results.csv"):
    print(f"\n🔬 Parsing Script: {file_path}")
    
    session_id, targets = parse_storyboard_for_eval(file_path)
    print(f"   Detected Session ID: {session_id}")
    print(f"   Found {len(targets)} total timeline events.")
    
    db = SQLLogger(session_id)
    headers = ["session_id", "tick", "room", "speaker", "dialogue", 
               "eval_2_verdict", "eval_2_reason", "eval_3_verdict", "eval_3_reason"]
    
    file_exists = os.path.isfile(output_csv)
    
    with open(output_csv, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(headers)
            
        for idx, target in enumerate(targets, 1):
            tick = target["tick"]
            speaker = target["speaker"]
            dialogue = target["dialogue"]
            room = target["room"]
            
            # -- Bypass LLM for Action-Only Ticks --
            if target.get("is_action"):
                writer.writerow([
                    session_id, tick, room, speaker, dialogue,
                    "N/A", "Deterministic engine action.",
                    "N/A", "Deterministic engine action."
                ])
                print(f"  [{idx}/{len(targets)}] Tick {tick} | Actions Only -> Skipped LLM")
                continue

            print(f"\n  [{idx}/{len(targets)}] Evaluating Tick {tick} | {speaker}...")
            
            # Fetch Ground Truth
            loc_logs = db.get_tick_locations(session_id, tick) or "None recorded."
            sight_logs = db.get_tick_sight(session_id, tick) or "None recorded."
            event_logs = get_rolling_events(db, session_id, tick)
            
            profile = db.get_npc_profile(speaker)
            persona_data = profile[0] if profile else f"You are {speaker}."
            
            # --- EVAL 2: THE PHYSICS JUDGE ---
            prompt_e2 = EVAL2_PHYSICS_PROMPT.format(
                LOC_LOGS=loc_logs, SIGHT_LOGS=sight_logs, EVENT_LOGS=event_logs,
                SPEAKER=speaker, DIALOGUE=dialogue
            )
            
            try:
                res_e2 = client.models.generate_content(
                    model=JUDGE_MODEL, contents=prompt_e2,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=Eval2Judgment, temperature=0.0 
                    )
                )
                e2_data = json.loads(res_e2.text)
            except Exception as e:
                print(f"     [!] Eval 2 Failed: {e}")
                e2_data = {"verdict": "ERROR", "reason": str(e)}

            # --- EVAL 3: THE PERSONA JUDGE ---
            prompt_e3 = EVAL3_PERSONA_PROMPT.format(
                PERSONA_DATA=persona_data,
                SPEAKER=speaker, DIALOGUE=dialogue
            )
            
            try:
                res_e3 = client.models.generate_content(
                    model=JUDGE_MODEL, contents=prompt_e3,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=Eval3Judgment, temperature=0.0 
                    )
                )
                e3_data = json.loads(res_e3.text)
            except Exception as e:
                print(f"     [!] Eval 3 Failed: {e}")
                e3_data = {"verdict": "ERROR", "reason": str(e)}

            # --- MERGE AND WRITE ---
            writer.writerow([
                session_id, tick, room, speaker, dialogue,
                e2_data.get("verdict"), e2_data.get("reason"),
                e3_data.get("verdict"), e3_data.get("reason")
            ])
            print(f"     ✅ E2: {e2_data.get('verdict')} | E3: {e3_data.get('verdict')}")

    print(f"\n📊 Evaluation complete. Results appended to {output_csv}")

if __name__ == "__main__":
    print("=========================================")
    print("🤖 DIRECT ENTAILMENT (NLI) EVALUATOR")
    print("=========================================")
    FILE_TARGET = input("Enter the path to the Storyboard .txt file: ").strip()
    
    if os.path.exists(FILE_TARGET):
        run_nli_on_script(FILE_TARGET)
    else:
        print("❌ File not found. Please check the path and try again.")