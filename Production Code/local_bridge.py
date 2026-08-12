import anvil.server
import anvil.media
import os
import io
import uuid
import zipfile
import time
from dotenv import load_dotenv

import chat_logger
from chat_logger import GameDBManager, SQLLogger, AutomatedDirector, SceneSelector
from condition_c import BaselineRunner
from tasks import (
    async_film_scene, 
    async_recap_job, 
    async_episode_job, 
    async_recast_character, 
    async_prepare_candidates_assets
)

load_dotenv()

OUTPUT_DIR = "Output_Videos"
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
if not os.path.exists("Assets"): os.makedirs("Assets")

local_sessions = {}
db_manager = GameDBManager()
director = AutomatedDirector()
GAMES_DIR = "games"
db_manager.sync_games_directory(GAMES_DIR)

# Connect to Anvil
anvil.server.connect(os.getenv("ANVIL_UPLINK_KEY", "YOUR_UPLINK_KEY_HERE"))
print("⚡ Connected to Cloud Gateway via Anvil Uplink! Listening for reviewer requests...")

@anvil.server.callable
def local_list_games():
    games = [f for f in os.listdir(GAMES_DIR) if f.endswith('.z8')]
    return {"available_games": games}

@anvil.server.callable
def local_start_game(game_filename, username, clear_db, gemini_key=None):
    if not game_filename: return {"error": "No game_file provided"}
    
    safe_filename = os.path.basename(game_filename)
    game_path = os.path.join(GAMES_DIR, safe_filename)
    if not os.path.exists(game_path): return {"error": "Game file not found."}

    safe_user = "".join(x for x in username if x.isalnum() or x in "-_") if username else "Guest"
    session_id = f"{safe_user}_{os.path.splitext(safe_filename)[0]}"
    
    if clear_db:
        try:
            logger = SQLLogger(session_id) 
            logger.supabase.table("chat_logs").delete().eq("session_id", session_id).execute()
            logger.supabase.table("session_meta").delete().eq("session_id", session_id).execute()
            logger.supabase.table("room_desc").delete().eq("session_id", session_id).execute()
            logger.supabase.table("official_timeline").delete().eq("session_id", session_id).execute()
        except Exception as e:
            print(f"⚠️ Could not delete cloud DB records: {e}")

    # Spin up the local engine
    controller = BaselineRunner(game_path, session_id)
    controller.game_name = safe_filename 
    controller.logger = controller.db 
    local_sessions[session_id] = controller
    
    return {
        "message": "Condition C Simulation active!", 
        "session_id": session_id,
        "tick": controller.tick_count,
        "game_used": safe_filename
    }

@anvil.server.callable
def local_step_game(session_id, gemini_key=None):
    controller = local_sessions.get(session_id)
    if not controller: return {"error": "Invalid session"}
    
    step_data = controller.step(api_key=gemini_key)
    return {
        "tick": controller.tick_count, 
        "locations": step_data["locations"],
        "agent_states": step_data.get("states", {})
    }

@anvil.server.callable
def local_orchestrate(session_id, seed, gemini_key=None):
    controller = local_sessions.get(session_id)
    if not controller: return {"error": "Invalid session"}
    if not seed: return {"error": "No narrative seed provided"}
    
    controller.orchestrate_narrative(seed, api_key=gemini_key)
    return {"message": f"Narrative seed injected: {seed}"}

@anvil.server.callable
def local_get_candidates(session_id, gemini_key=None, gcp_key=None):
    controller = local_sessions.get(session_id)
    if not controller: return {"error": "Invalid session"}
    
    video_logger = controller.logger
    videomaker = SceneSelector(db=video_logger, director=director, key_terms=["darkness", "grue", "lamp", "spell"], game_name=controller.game_name)
    top_scenes = videomaker.scan_and_rank(start_tick=0, end_tick=controller.tick_count)
    
    raw_candidates = []
    scenes_for_worker = []

    for score, tick, room, scene_data in top_scenes:
        filtered_script = [line for line in scene_data["script"] if line["speaker"].upper() not in ["ACTION", "SYSTEM"]]
        if not filtered_script: continue
            
        scene_data["script"] = filtered_script
        scenes_for_worker.append(scene_data)
        raw_candidates.append({"tick": tick, "room": room, "score": score, "script": scene_data["script"]})
        
    async_prepare_candidates_assets(session_id, controller.game_name, scenes_for_worker, gemini_key, gcp_key)
    return {"candidates": raw_candidates}

@anvil.server.callable
def local_queue_render(session_id, tick, room_name, mode, gemini_key=None, gcp_key=None):
    controller = local_sessions.get(session_id)
    if not controller: return {"error": "Invalid session"}

    video_logger = controller.logger
    scene_data = video_logger.get_structured_scene_data(tick, room_name)
    scene_data["script"] = [line for line in scene_data["script"] if line["speaker"].upper() not in ["ACTION", "SYSTEM"]]
    if not scene_data["script"]: return {"error": "No dialogue found."}
    
    agents = []
    speakers = set()
    formatted_script = []
    directions = ["north", "east", "south", "west"]
    videomaker = SceneSelector(db=video_logger, director=director, key_terms=[], game_name=controller.game_name)

    for line in scene_data["script"]:
        speaker = line["speaker"]
        emotion = videomaker._detect_emotion_nrc(line["line"]) 
        voice_id = video_logger.get_npc_voice(speaker)
        formatted_script.append((speaker, line["line"], emotion, voice_id))
        if speaker not in speakers:
            assigned_wall = directions[len(speakers) % 4]
            agents.append({"name": speaker, "desc": "A character", "facing": assigned_wall})
            speakers.add(speaker)

    safe_room = room_name.replace(" ", "_").replace("'", "")
    prefix = "animatic" if mode == "stills" else "final_render"
    filename = f"{prefix}_{controller.game_name}_{safe_room}_tick{tick}.mp4"
    output_path = os.path.join(OUTPUT_DIR, filename)
    script_path = output_path.rsplit('.', 1)[0] + '.txt'
    
    try:
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(f"LOCATION: {room_name}\nTIME: Tick {tick}\n\nSCENE DESCRIPTION:\n{scene_data['visual']}\n\n")
            if speakers:
                f.write("CAST PROFILES:\n")
                for spk in speakers:
                    prof = video_logger.get_npc_profile(spk)
                    f.write(f"- {spk}: {prof[0] if prof else 'No profile.'}\n")
                f.write("\n")
            f.write("-----------------------------------------\n")
            for speaker, text, emotion, _ in formatted_script:
                f.write(f"{speaker}: *{emotion.upper()}* {text.strip()}\n")
    except Exception as e:
        print(f"⚠️ Failed to save rich transcript: {e}")
        
    async_film_scene(
        scene_data["visual"], agents, formatted_script, 
        controller.game_name, room_name, output_path, mode,
        gemini_key, gcp_key
    )
    video_logger.close()
    return {"message": f"Render task queued for Tick {tick}!", "video_filename": filename}

@anvil.server.callable
def local_add_timeline(session_id, tick, room):
    controller = local_sessions.get(session_id)
    if not controller: return {"error": "Invalid session"}
    video_logger = controller.logger
    success = video_logger.add_to_timeline(tick, room)
    video_logger.close()
    return {"message": f"Added Tick {tick} to Timeline!"} if success else {"error": "Already in timeline"}

@anvil.server.callable
def local_queue_recap(session_id, past_n, gemini_key=None, gcp_key=None):
    controller = local_sessions.get(session_id)
    if not controller: return {"error": "Invalid session"}
    filename = f"Recap_{controller.game_name}_{uuid.uuid4().hex[:6]}.mp4"
    output_path = os.path.join(OUTPUT_DIR, filename)
    async_recap_job(session_id, controller.game_name, past_n, output_path, gemini_key, gcp_key)
    return {"message": f"Generating recap of last {past_n} events...", "video_filename": filename}

@anvil.server.callable
def local_get_timeline(session_id):
    controller = local_sessions.get(session_id)
    if not controller: return {"error": "Invalid session"}
    video_logger = controller.logger
    timeline_data = video_logger.get_official_timeline(100) 
    videomaker = SceneSelector(db=video_logger, director=director, key_terms=[], game_name=controller.game_name)
    
    for scene in timeline_data:
        scene['script'] = [line for line in scene['script'] if line['speaker'].upper() not in ["ACTION", "SYSTEM"]]
        scene_profiles = {}
        speakers = set(line['speaker'] for line in scene['script'])
        for spk in speakers:
            prof = video_logger.get_npc_profile(spk)
            if prof: scene_profiles[spk] = prof[0]
        scene['profiles'] = scene_profiles
        for line in scene['script']:
            line['emotion'] = videomaker._detect_emotion_nrc(line['line'])
            
    video_logger.close()
    return {"timeline": timeline_data}

@anvil.server.callable
def local_queue_episode(session_id, mode, gemini_key=None, gcp_key=None):
    controller = local_sessions.get(session_id)
    if not controller: return {"error": "Invalid session"}
    video_logger = controller.logger
    timeline_data = video_logger.get_official_timeline(100)
    if not timeline_data: return {"error": "Timeline is empty. Pin scenes first!"}

    prefix = "Animatic_Episode" if mode == "stills" else "Episode_Full"
    episode_filename = f"{prefix}_{controller.game_name}_{uuid.uuid4().hex[:6]}.mp4"
    output_path = os.path.join(OUTPUT_DIR, episode_filename)

    async_episode_job(session_id, controller.game_name, timeline_data, output_path, mode, gemini_key, gcp_key)
    return {"message": f"{prefix.replace('_', ' ')} compilation started!", "video_filename": episode_filename}

@anvil.server.callable
def local_get_cast(session_id):
    controller = local_sessions.get(session_id)
    if not controller: return {"error": "Invalid session"}
    npcs = controller.logger.get_all_npcs()
    cast_list = []
    for npc in npcs:
        agent_path = os.path.join("Assets", "Global_Agents", f"{npc['name']}.png")
        if not os.path.exists(agent_path):
            director.create_agent_plate(npc['name'], npc['appearance'])
        cache_buster = int(time.time())
        image_url = f"/Assets/Global_Agents/{npc['name']}.png?t={cache_buster}" if os.path.exists(agent_path) else None
        cast_list.append({"name": npc['name'], "appearance": npc['appearance'], "image_url": image_url})
    return {"cast": cast_list}

@anvil.server.callable
def local_queue_recast(session_id, npc_name, custom_prompt, gemini_key=None, gcp_key=None):
    if not npc_name or not custom_prompt: return {"error": "Missing name or prompt"}
    controller = local_sessions.get(session_id)
    if not controller: return {"error": "Invalid session"}
    async_recast_character(session_id, npc_name, custom_prompt, gemini_key, gcp_key)
    return {"message": f"Recast task queued for {npc_name}!", "status": "processing"}

@anvil.server.callable
def local_remove_timeline(session_id, tick, room):
    controller = local_sessions.get(session_id)
    if not controller: return {"error": "Invalid session"}
    video_logger = controller.logger
    success = video_logger.remove_from_timeline(tick, room)
    return {"message": f"Removed Tick {tick}"} if success else {"error": "Failed to remove"}

@anvil.server.callable
def local_upload_cast_image(session_id, npc_name, media_obj):
    if not npc_name: return {"error": "Missing params"}
    agent_dir = os.path.join("Assets", "Global_Agents")
    os.makedirs(agent_dir, exist_ok=True)
    filepath = os.path.join(agent_dir, f"{npc_name}.png")
    try:
        # Write the incoming Anvil Media Bytes directly to the PC
        with open(filepath, "wb") as f:
            f.write(media_obj.get_bytes())
        safe_url = filepath.replace("\\", "/")
        return {"message": f"Updated {npc_name}", "image_url": f"/{safe_url}?t={int(time.time())}"}
    except Exception as e:
        return {"error": f"Failed to save image: {e}"}

@anvil.server.callable
def local_spice_timeline(session_id, gemini_key=None):
    if gemini_key: chat_logger.client = genai.Client(api_key=gemini_key)
    controller = local_sessions.get(session_id)
    if not controller: return {"error": "Invalid session"}
    video_logger = controller.logger
    timeline_data = video_logger.get_official_timeline(100) 
    if not timeline_data: return {"error": "Timeline empty"}
    for scene in timeline_data:
        scene['script'] = [line for line in scene['script'] if line['speaker'].upper() not in ["ACTION", "SYSTEM"]]
    return director.spice_up_story(timeline_data, video_logger)

# ==========================================
# FILE STREAMING OVER ANVIL
# ==========================================

@anvil.server.callable
def local_render_status(filename):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(file_path): 
        return {"status": "done", "video_url": f"/videos/{filename}"}
    return {"status": "processing"}

@anvil.server.callable
def local_fetch_asset(filename):
    """Reads a local asset and wraps it in an Anvil Media Object."""
    path = os.path.join('Assets', filename)
    if not os.path.exists(path): return None
    with open(path, 'rb') as f:
        return anvil.BlobMedia('image/png', f.read(), name=filename)

@anvil.server.callable
def local_fetch_video(filename):
    """Reads a local video and wraps it in an Anvil Media Object."""
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path): return None
    with open(path, 'rb') as f:
        return anvil.BlobMedia('video/mp4', f.read(), name=filename)

@anvil.server.callable
def local_export_eval_kit(session_id):
    """Generates the evaluation ZIP locally and passes it back over the tunnel."""
    controller = local_sessions.get(session_id)
    if not controller: return None
    video_logger = controller.logger
    timeline_data = video_logger.get_official_timeline(100)
    
    transcript_text = f"=== FULL EPISODE TRANSCRIPT: {controller.game_name} ===\nSession ID: {session_id}\n\n"
    for scene in timeline_data:
        transcript_text += f"LOCATION: {scene['room']}\nTIME: Tick {scene['tick']}\n\n"
        script_lines = [line for line in scene['script'] if line['speaker'].upper() not in ["ACTION", "SYSTEM"]]
        for line in script_lines: transcript_text += f"{line['speaker']}: {line['line']}\n"
        transcript_text += "\n-----------------------------------------\n\n"

    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        transcript_filename = f"{session_id}_transcript.txt"
        zf.writestr(f"aiide_evaluation_kit/{transcript_filename}", transcript_text)
        for script in ["dramabench_eval.py", "nli_benchmark.py", "stubbornness_test.py"]:
            if os.path.exists(script): zf.write(script, arcname=f"aiide_evaluation_kit/{script}")
                
    memory_file.seek(0)
    return anvil.BlobMedia('application/zip', memory_file.read(), name=f"AIIDE_Eval_Kit_{session_id}.zip")

# Keep the PC listening for commands from the cloud
anvil.server.wait_forever()