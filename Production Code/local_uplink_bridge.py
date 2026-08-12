# local_uplink_bridge.py
import anvil.server
import os
from dotenv import load_dotenv

# Import your existing engine and worker dispatchers
import chat_logger
from condition_c_3 import BaselineRunner
from tasks import (
    async_film_scene, 
    async_recap_job, 
    async_episode_job, 
    async_recast_character, 
    async_prepare_candidates_assets
)

load_dotenv()

# Active session instances in local PC memory
local_sessions = {}

# Connect to your Anvil Cloud App using your Uplink Key
ANVIL_UPLINK_KEY = os.getenv("ANVIL_UPLINK_KEY", "YOUR-UPLINK-KEY-HERE")
anvil.server.connect(ANVIL_UPLINK_KEY)
print("⚡ Connected to Anvil Cloud Gateway! Listening for reviewer requests...")

@anvil.server.callable
def local_start_game(game_filename, username, clear_db, gemini_key=None):
    safe_filename = os.path.basename(game_filename)
    game_path = os.path.join("games", safe_filename)
    safe_user = "".join(x for x in username if x.isalnum() or x in "-_")
    session_id = f"{safe_user}_{os.path.splitext(safe_filename)[0]}"

    runner = BaselineRunner(game_path, session_id)
    runner.game_name = safe_filename
    runner.logger = runner.db
    local_sessions[session_id] = runner

    return {
        "session_id": session_id,
        "tick": runner.tick_count,
        "game_used": safe_filename
    }

@anvil.server.callable
def local_step_game(session_id, gemini_key=None):
    runner = local_sessions.get(session_id)
    if not runner:
        # Re-instantiate from Supabase cloud save if process restarted
        game_path = os.path.join("games", "TheCipherBlackSite1.z8")
        runner = BaselineRunner(game_path, session_id)
        local_sessions[session_id] = runner

    step_data = runner.step(api_key=gemini_key)
    return {
        "tick": runner.tick_count,
        "locations": step_data["locations"],
        "agent_states": step_data.get("states", {})
    }

@anvil.server.callable
def local_orchestrate(session_id, seed, gemini_key=None):
    runner = local_sessions.get(session_id)
    if runner:
        runner.orchestrate_narrative(seed, api_key=gemini_key)
        return {"status": "success"}
    return {"status": "error", "message": "Session not active"}

@anvil.server.callable
def local_queue_render(session_id, tick, room_name, mode, gemini_key=None, gcp_key=None):
    runner = local_sessions.get(session_id)
    if not runner: return {"error": "Invalid session"}

    video_logger = runner.logger
    scene_data = video_logger.get_structured_scene_data(tick, room_name)
    scene_data["script"] = [l for l in scene_data["script"] if l["speaker"].upper() not in ["ACTION", "SYSTEM"]]
    
    # ... (Prepare formatted_script and agents) ...
    
    safe_room = room_name.replace(" ", "_").replace("'", "")
    prefix = "animatic" if mode == "stills" else "final_render"
    filename = f"{prefix}_{runner.game_name}_{safe_room}_tick{tick}.mp4"
    output_path = os.path.join("Output_Videos", filename)

    # Hand off to Huey local process immediately
    async_film_scene(
        scene_data["visual"], agents, formatted_script, 
        runner.game_name, room_name, output_path, mode,
        gemini_key, gcp_key
    )
    video_logger.close()

    return {"video_filename": filename}

@anvil.server.callable
def local_queue_recast(session_id, npc_name, custom_prompt, gemini_key=None, gcp_key=None):
    async_recast_character(session_id, npc_name, custom_prompt, gemini_key, gcp_key)
    return {"status": "queued"}

# Keep Uplink server waiting for calls
anvil.server.wait_forever()