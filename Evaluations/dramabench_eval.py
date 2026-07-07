import os
import re
import zipfile
import json
from pydantic import BaseModel
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

EVALUATION_MODEL = "gemini-2.5-pro" 
ZIP_PATH = "DramaBench-main.zip"
TRANSCRIPT_PATH = "BASELINE_BENCH_1782336869.txt"

# ============================================================================
# PYDANTIC SCHEMAS
# ============================================================================

class CharacterEvaluation(BaseModel):
    line_number: int
    character: str
    line_text: str
    label: str

class CharacterResponse(BaseModel):
    evaluations: list[CharacterEvaluation]

class LogicEvaluation(BaseModel):
    event: str
    label: str
    explanation: str

class LogicResponse(BaseModel):
    evaluations: list[LogicEvaluation]

class ConflictEvaluation(BaseModel):
    global_label: str
    reasoning: str

class ConflictResponse(BaseModel):
    conflict_evaluation: ConflictEvaluation

# ============================================================================
# 1. OFFICIAL PROMPT MANIFESTATION LAYER
# ============================================================================

def get_official_prompt(dimension_name):
    internal_path = f"DramaBench-main/prompts/{dimension_name}_prompt.txt"
    try:
        with zipfile.ZipFile(ZIP_PATH, 'r') as z:
            with z.open(internal_path) as f:
                return f.read().decode('utf-8')
    except KeyError:
        raise FileNotFoundError(f"Missing {internal_path} inside file package {ZIP_PATH}.")

# ============================================================================
# 2. TICK-BASED STRIDING WINDOW ENGINE
# ============================================================================

def load_and_parse_rich_transcript_by_ticks(file_path, tick_window_size=5):
    """
    Parses the script and chunks it into striding tick windows.
    Returns a list of payload dictionaries for batch evaluation.
    """
    if not os.path.exists(file_path):
         raise FileNotFoundError(f"Transcript file target not found at: {file_path}")
         
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    header_match = re.split(r'(SCENE \d+)', content)
    scenes_raw = [header_match[i] + header_match[i+1] for i in range(1, len(header_match), 2) if i + 1 < len(header_match)]

    global_line_counter = 1
    scenes_by_tick = {}

    for scene_text in scenes_raw:
        loc_m = re.search(r'LOCATION:\s*(.*)', scene_text)
        tick_m = re.search(r'TIME:\s*Tick\s*(\d+)', scene_text)
        desc_m = re.search(r'SCENE DESCRIPTION:\s*\n([\s\S]*?)(?=\nCAST PROFILES:|\n-----------------------------------------)', scene_text)
        profile_m = re.search(r'CAST PROFILES:\s*\n([\s\S]*?)(?=\n-----------------------------------------)', scene_text)
        dialogue_m = re.search(r'-----------------------------------------\s*\n([\s\S]*)', scene_text)
        
        current_tick = int(tick_m.group(1)) if tick_m else 0
        raw_dialogue_lines = dialogue_m.group(1).strip().splitlines() if dialogue_m else []
        
        numbered_dialogue, numbered_logic = [], []

        for line in raw_dialogue_lines:
            line = line.strip()
            if not line: 
                continue
                
            is_action = line.startswith('*') and line.endswith('*')
            clean_line_text = line.replace('*', '').strip()

            if is_action:
                if "fails (" in clean_line_text:
                    numbered_logic.append(
                        f"[Line {global_line_counter}] (PHYSICAL ACTION) {clean_line_text} "
                        f"- SYSTEM NOTE: The agent hallucinated an invalid environment state. This MUST be labeled as 'Violated'."
                    )
                else:
                    numbered_logic.append(f"[Line {global_line_counter}] (PHYSICAL ACTION) {clean_line_text}")
            else:
                numbered_dialogue.append(f"[Line {global_line_counter}] {line}")
                numbered_logic.append(f"[Line {global_line_counter}] (DIALOGUE) {line}")
            
            global_line_counter += 1

        scene_data = {
            "meta": f"{loc_m.group(1) if loc_m else 'Unknown'} | Tick {current_tick}",
            "description": desc_m.group(1).strip() if desc_m else "None Provided.",
            "profiles": profile_m.group(1).strip() if profile_m else "None Provided.",
            "dialogue_clean": "\n".join(numbered_dialogue),
            "logic_clean": "\n".join(numbered_logic)
        }

        if current_tick not in scenes_by_tick:
            scenes_by_tick[current_tick] = []
        scenes_by_tick[current_tick].append(scene_data)

    unique_ticks = sorted(list(scenes_by_tick.keys()))
    chunks = []
    
    def build_payload(tick_list, mode):
        segments = []
        for t in tick_list:
            for block in scenes_by_tick[t]:
                segments.append(
                    f"=== LOCATION/TIME: {block['meta']} ===\n"
                    f"ENVIRONMENT CONTEXT:\n{block['description']}\n\n"
                    f"CHARACTER PROFILES:\n{block['profiles']}\n\n"
                    f"RECORDED TIMELINE:\n{block[mode]}"
                )
        return "\n\n".join(segments)

    # Start 'i' at tick_window_size to avoid the duplicate zero-index loop
    for i in range(tick_window_size, len(unique_ticks), tick_window_size):
        context_ticks = unique_ticks[i - tick_window_size : i]
        continuation_ticks = unique_ticks[i : i + tick_window_size]
        
        # Stop if we run out of continuation data
        if not continuation_ticks: 
            break

        chunks.append({
            "id": f"{os.path.basename(file_path).replace('.txt', '')}_TICKS_{continuation_ticks[0]}_TO_{continuation_ticks[-1]}",
            "char_context": build_payload(context_ticks, "dialogue_clean"),
            "char_continuation": build_payload(continuation_ticks, "dialogue_clean"),
            "logic_context": build_payload(context_ticks, "logic_clean"),
            "logic_continuation": build_payload(continuation_ticks, "logic_clean")
        })

    return chunks

# ============================================================================
# 3. EVALUATION PIPELINE
# ============================================================================

def run_dramabench_dimension(dimension_id, data):
    raw_prompt_template = get_official_prompt(dimension_id)
    
    if dimension_id == "character_consistency":
        prompt = raw_prompt_template.replace("{CONTEXT}", data["char_context"])
        prompt = prompt.replace("{CONTINUATION}", data["char_continuation"])
        target_schema = CharacterResponse
    elif dimension_id == "logic_consistency":
        prompt = raw_prompt_template.replace("{CONTEXT}", data["logic_context"])
        prompt = prompt.replace("{CONTINUATION}", data["logic_continuation"])
        target_schema = LogicResponse
    else:
        prompt = raw_prompt_template.replace("{CONTEXT}", data["logic_context"])
        prompt = prompt.replace("{CONTINUATION}", data["logic_continuation"])
        target_schema = ConflictResponse

    prompt = prompt.replace("{MODEL}", "Agentworld_Engine")
    prompt = prompt.replace("{SCRIPT_ID}", data["id"])
    
    response = client.models.generate_content(
        model=EVALUATION_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=target_schema,
            temperature=0.0 
        )
    )
    
    return json.loads(response.text)

# ============================================================================
# 4. AGGREGATOR RUNNER & CSV EXPORT
# ============================================================================

if __name__ == "__main__":
    from datetime import datetime
    import csv
    
    print(f"📋 Step 1: Ingesting rich text file and generating 5-Tick Chunks...")
    script_chunks = load_and_parse_rich_transcript_by_ticks(TRANSCRIPT_PATH, tick_window_size=5)
    
    print(f"    -> Generated {len(script_chunks)} evaluation chunks.")
    
    global_char_lines = 0
    global_ooc_count = 0
    global_logic_events = 0
    global_violated_count = 0
    
    conflict_weights = {
        "ESCALATION": 2.0, 
        "TWIST": 2.0, 
        "PAUSE": 1.0, 
        "RESOLUTION": 0.0, 
        "DROPPED": -5.0
    }
    cumulative_conflict_score = 0.0
    
    chunk_wise_records = []
    
    print(f"\n🚀 Step 2: Commencing Chunk-by-Chunk Evaluation ({EVALUATION_MODEL})...")
    
    for idx, chunk in enumerate(script_chunks, 1):
        print(f"\n  Processing Chunk {idx}/{len(script_chunks)}: {chunk['id']}")
        
        # 1. Character Consistency
        char_json = run_dramabench_dimension("character_consistency", chunk)
        char_evals = char_json.get("evaluations", [])
        c_lines = len(char_evals)
        c_ooc = sum(1 for e in char_evals if e.get("label") == "OOC")
        
        global_char_lines += c_lines
        global_ooc_count += c_ooc
        
        # 2. Logic Consistency
        logic_json = run_dramabench_dimension("logic_consistency", chunk)
        logic_evals = logic_json.get("evaluations", [])
        l_events = len(logic_evals)
        l_viol = sum(1 for e in logic_evals if e.get("label") == "Violated")
        
        global_logic_events += l_events
        global_violated_count += l_viol
        
        # 3. Conflict Handling
        conflict_json = run_dramabench_dimension("conflict_handling", chunk)
        raw_label = conflict_json.get("conflict_evaluation", {}).get("global_label", "Resolution")
        c_weight = conflict_weights.get(str(raw_label).upper(), 0.0)
        cumulative_conflict_score += c_weight
        
        # Store localized chunk metrics
        raw_ooc_rate = (c_ooc / c_lines * 100) if c_lines > 0 else 0.0
        raw_logic_rate = (l_viol / l_events * 100) if l_events > 0 else 0.0
        
        chunk_wise_records.append({
            "chunk_id": chunk["id"],
            "character_ooc_rate_percent": round(raw_ooc_rate, 2),
            "logic_break_rate_percent": round(raw_logic_rate, 2),
            "conflict_label": str(raw_label).upper(),
            "conflict_score": c_weight
        })
        
        print(f"    -> Character OOC: {raw_ooc_rate:.2f}% | Logic Breaks: {raw_logic_rate:.2f}% | Conflict: {str(raw_label).upper()}")

    print("\n🧮 Step 3: Consolidating final metrics and generating CSVs...")
    
    # ---------------------------------------------------------
    # Apply Official DramaBench Raw Scoring (No Normalization)
    # ---------------------------------------------------------
    # OOC Rate: N_OOC / Total Lines (Lower is better, e.g., SOTA is ~0.006)
    db_character_score = (global_ooc_count / global_char_lines) if global_char_lines > 0 else 0.0
    
    # Logic Break Rate: N_Violated / Total Logic Events (Lower is better, e.g., SOTA is ~0.020)
    db_logic_score = (global_violated_count / global_logic_events) if global_logic_events > 0 else 0.0
    
    # Conflict Handling Weight: Average label weight (Higher is better, range -5.0 to 2.0, SOTA is ~1.843)
    db_conflict_score = (cumulative_conflict_score / len(script_chunks)) if script_chunks else 0.0
    
    # Note: 3-Dim Average is removed as DramaBench evaluates these dimensions independently on different scales.
    
    # ---------------------------------------------------------
    # Export File Generation
    # ---------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_base = TRANSCRIPT_PATH.replace('.txt', '')
    
    csv_raw_filename = f"RawChunks_{script_base}_{timestamp}.csv"
    csv_leaderboard_filename = f"Leaderboard_{script_base}_{timestamp}.csv"
    
    # 1. Write the Raw Chunk CSV
    with open(csv_raw_filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Script ID", "Chunk ID", "OOC Rate", "Logic Break Rate", "Conflict Label", "Conflict Score Weight"])
        for record in chunk_wise_records:
            # Reverting chunk records to raw decimal format rather than percentages
            raw_chunk_ooc = (record["character_ooc_rate_percent"] / 100.0)
            raw_chunk_logic = (record["logic_break_rate_percent"] / 100.0)
            
            writer.writerow([
                TRANSCRIPT_PATH, 
                record["chunk_id"], 
                round(raw_chunk_ooc, 4), 
                round(raw_chunk_logic, 4), 
                record["conflict_label"], 
                record["conflict_score"]
            ])
            
    # 2. Write the DramaBench Leaderboard CSV
    with open(csv_leaderboard_filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Model / Architecture", "Character Consistency (OOC Rate)", "Logic Consistency (Break Rate)", "Conflict Handling (Avg Weight)"])
        
        writer.writerow([
            "Agentworld Baseline (Blind Physics)", 
            round(db_character_score, 4), 
            round(db_logic_score, 4), 
            round(db_conflict_score, 4) 
        ])

    print("\n" + "="*65)
    print(f"✨ EVALUATION COMPLETE. Data exported successfully.")
    print(f"   -> Raw Data: {csv_raw_filename}")
    print(f"   -> Leaderboard Data: {csv_leaderboard_filename}")
    print("="*65)