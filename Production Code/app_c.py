import os
import uuid
import io
import zipfile
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from google import genai

import chat_logger
from chat_logger import GameDBManager, SQLLogger, AutomatedDirector, SceneSelector
from condition_c import BaselineRunner
from tasks import async_film_scene, async_recap_job, async_episode_job, async_recast_character, async_prepare_candidates_assets
import time
from PIL import Image

OUTPUT_DIR = "Output_Videos"
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
if not os.path.exists("Assets"): os.makedirs("Assets")

app = Flask(__name__)
CORS(app)

active_sessions = {}
db_manager = GameDBManager()
director = AutomatedDirector()

GAMES_DIR = "games"
db_manager.sync_games_directory(GAMES_DIR)

def get_reviewer_keys():
    """Extracts custom reviewer keys from HTTP headers."""
    gemini_key = request.headers.get('X-Gemini-Key')
    gcp_key = request.headers.get('X-GCP-Key')
    return gemini_key, gcp_key

@app.route('/')
def serve_dashboard():
    return send_from_directory('.', 'index.html')
    
@app.route('/Assets/<path:filename>')
def serve_assets(filename):
    return send_from_directory('Assets', filename)

@app.route('/games', methods=['GET'])
def list_games():
    games = [f for f in os.listdir(GAMES_DIR) if f.endswith('.z8')]
    return jsonify({"available_games": games})

@app.route('/game/start', methods=['POST'])
def start_game():
    data = request.json
    game_filename = data.get("game_file")
    username = data.get("username", "Guest").strip()
    clear_db = data.get("clear_db", False)
    
    if not game_filename:
         return jsonify({"error": "No game_file provided in request"}), 400
    if not username:
         username = "Guest"
    
    safe_filename = os.path.basename(game_filename)
    game_path = os.path.join(GAMES_DIR, safe_filename)
    
    if not os.path.exists(game_path):
        return jsonify({"error": f"Game file '{safe_filename}' not found."}), 404

    safe_user = "".join(x for x in username if x.isalnum() or x in "-_")
    session_id = f"{safe_user}_{os.path.splitext(safe_filename)[0]}"
    
    if clear_db:
        print(f"🗑️ Wiping previous cloud database records for {session_id}...")
        try:
            logger = SQLLogger(session_id) 
            logger.supabase.table("chat_logs").delete().eq("session_id", session_id).execute()
            logger.supabase.table("session_meta").delete().eq("session_id", session_id).execute()
            logger.supabase.table("room_desc").delete().eq("session_id", session_id).execute()
            logger.supabase.table("official_timeline").delete().eq("session_id", session_id).execute()
            print("✅ Cloud DB wipe successful!")
        except Exception as e:
            print(f"⚠️ Could not delete cloud DB records: {e}")

    controller = BaselineRunner(game_path, session_id)
    controller.game_name = safe_filename 
    controller.logger = controller.db 
    
    active_sessions[session_id] = controller
    
    return jsonify({
        "message": f"Welcome back, {username}!" if controller.tick_count > 0 else f"Condition C Simulation active for {username}!", 
        "session_id": session_id,
        "tick": controller.tick_count,
        "game_used": safe_filename
    })

@app.route('/game/<session_id>/step', methods=['POST'])
def step_game(session_id):
    gemini_key, _ = get_reviewer_keys()
    controller = active_sessions.get(session_id)
    if not controller: return jsonify({"error": "Invalid session"}), 404
    
    step_data = controller.step(api_key=gemini_key)
    
    return jsonify({
        "tick": controller.tick_count, 
        "locations": step_data["locations"],
        "agent_states": step_data.get("states", {})
    })

@app.route('/game/<session_id>/orchestrate', methods=['POST'])
def orchestrate_narrative(session_id):
    gemini_key, _ = get_reviewer_keys()
    data = request.json
    seed = data.get("seed")
    
    controller = active_sessions.get(session_id)
    if not controller: return jsonify({"error": "Invalid session"}), 404
    if not seed: return jsonify({"error": "No narrative seed provided"}), 400
    
    controller.orchestrate_narrative(seed, api_key=gemini_key)
    return jsonify({"message": f"Narrative seed injected: {seed}"})

@app.route('/game/<session_id>/candidates', methods=['POST'])
def get_candidates(session_id):
    gemini_key, gcp_key = get_reviewer_keys()
    controller = active_sessions.get(session_id)
    if not controller: return jsonify({"error": "Invalid session"}), 404
    
    video_logger = controller.logger
    game_terms = ["darkness", "grue", "lamp", "spell"]
    videomaker = SceneSelector(db=video_logger, director=director, key_terms=game_terms, game_name=controller.game_name)
    
    top_scenes = videomaker.scan_and_rank(start_tick=0, end_tick=controller.tick_count)
    
    raw_candidates = []
    scenes_for_worker = []

    for score, tick, room, scene_data in top_scenes:
        filtered_script = [line for line in scene_data["script"] if line["speaker"].upper() not in ["ACTION", "SYSTEM"]]
        if not filtered_script: 
            continue
            
        scene_data["script"] = filtered_script
        scenes_for_worker.append(scene_data)
        
        raw_candidates.append({
            "tick": tick,
            "room": room,
            "score": score,
            "script": scene_data["script"]
        })
        
    async_prepare_candidates_assets(session_id, controller.game_name, scenes_for_worker, gemini_key, gcp_key)
        
    return jsonify({"candidates": raw_candidates})

@app.route('/videos/<path:filename>')
def serve_videos(filename):
    return send_from_directory(OUTPUT_DIR, filename)

@app.route('/render_status/<filename>', methods=['GET'])
def render_status(filename):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(file_path):
        return jsonify({"status": "done", "video_url": f"/videos/{filename}"})
    else:
        return jsonify({"status": "processing"})

@app.route('/game/<session_id>/render', methods=['POST'])
def render_video(session_id):
    gemini_key, gcp_key = get_reviewer_keys()
    data = request.json
    tick, room_name = data.get("tick"), data.get("room")
    mode = data.get("mode", "full")
    
    controller = active_sessions.get(session_id)
    if not controller: return jsonify({"error": "Invalid session"}), 404

    video_logger = controller.logger
    scene_data = video_logger.get_structured_scene_data(tick, room_name)
    scene_data["script"] = [line for line in scene_data["script"] if line["speaker"].upper() not in ["ACTION", "SYSTEM"]]
    
    if not scene_data["script"]:
        return jsonify({"error": "No dialogue found in this scene to render."}), 400
    
    agents = []
    speakers = set()
    formatted_script = []
    directions = ["north", "east", "south", "west"]

    videomaker = SceneSelector(db=video_logger, director=director, key_terms=[], game_name=controller.game_name)

    for line in scene_data["script"]:
        speaker = line["speaker"]
        text = line["line"]
        emotion = videomaker._detect_emotion_nrc(line["line"]) 
        voice_id = video_logger.get_npc_voice(speaker)
        formatted_script.append((speaker, text, emotion, voice_id))
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
    return jsonify({
        "message": f"Render task queued for Tick {tick}!", 
        "video_filename": filename 
    })

@app.route('/game/<session_id>/recap', methods=['POST'])
def create_recap(session_id):
    gemini_key, gcp_key = get_reviewer_keys()
    data = request.json
    past_n = data.get("past_n", 3)
    
    controller = active_sessions.get(session_id)
    if not controller: return jsonify({"error": "Invalid session"}), 404

    filename = f"Recap_{controller.game_name}_{uuid.uuid4().hex[:6]}.mp4"
    output_path = os.path.join(OUTPUT_DIR, filename)

    async_recap_job(session_id, controller.game_name, past_n, output_path, gemini_key, gcp_key)
    
    return jsonify({
        "message": f"Generating recap of last {past_n} events...",
        "video_filename": filename
    })

@app.route('/game/<session_id>/episode/render', methods=['POST'])
def render_full_episode(session_id):
    gemini_key, gcp_key = get_reviewer_keys()
    data = request.json or {}
    mode = data.get("mode", "full")
    
    controller = active_sessions.get(session_id)
    if not controller: return jsonify({"error": "Invalid session"}), 404

    video_logger = controller.logger
    timeline_data = video_logger.get_official_timeline(100)
    
    if not timeline_data:
        return jsonify({"error": "Timeline is empty. Pin some scenes first!"}), 400

    prefix = "Animatic_Episode" if mode == "stills" else "Episode_Full"
    episode_filename = f"{prefix}_{controller.game_name}_{uuid.uuid4().hex[:6]}.mp4"
    output_path = os.path.join(OUTPUT_DIR, episode_filename)

    async_episode_job(session_id, controller.game_name, timeline_data, output_path, mode, gemini_key, gcp_key)
    
    return jsonify({
        "message": f"{prefix.replace('_', ' ')} compilation started!", 
        "video_filename": episode_filename 
    })

@app.route('/game/<session_id>/recast', methods=['POST'])
def recast_character(session_id):
    gemini_key, gcp_key = get_reviewer_keys()
    data = request.json
    npc_name = data.get("npc_name")
    custom_prompt = data.get("custom_prompt")
    
    if not npc_name or not custom_prompt:
        return jsonify({"error": "Missing name or prompt"}), 400
        
    controller = active_sessions.get(session_id)
    if not controller: 
        return jsonify({"error": "Invalid session. Please start a game first."}), 404
    
    async_recast_character(session_id, npc_name, custom_prompt, gemini_key, gcp_key)
    
    return jsonify({
        "message": f"Recast task queued for {npc_name}!", 
        "status": "processing"
    })

@app.route('/game/<session_id>/timeline/add', methods=['POST'])
def add_to_timeline(session_id):
    data = request.json
    tick, room = data.get("tick"), data.get("room")
    controller = active_sessions.get(session_id)
    if not controller: return jsonify({"error": "Invalid session"}), 404

    video_logger = controller.logger
    success = video_logger.add_to_timeline(tick, room)
    video_logger.close()
    return jsonify({"message": f"Added Tick {tick} to Timeline!"}) if success else jsonify({"error": "Already in timeline"}), 400

@app.route('/game/<session_id>/timeline', methods=['GET'])
def get_timeline(session_id):
    controller = active_sessions.get(session_id)
    if not controller: return jsonify({"error": "Invalid session"}), 404

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
    return jsonify({"timeline": timeline_data})

@app.route('/game/<session_id>/cast', methods=['GET'])
def get_cast(session_id):
    controller = active_sessions.get(session_id)
    if not controller: return jsonify({"error": "Invalid session"}), 404
    
    npcs = controller.logger.get_all_npcs()
    cast_list = []
    for npc in npcs:
        agent_path = os.path.join("Assets", "Global_Agents", f"{npc['name']}.png")
        if not os.path.exists(agent_path):
            director.create_agent_plate(npc['name'], npc['appearance'])
        cache_buster = int(time.time())
        image_url = f"/Assets/Global_Agents/{npc['name']}.png?t={cache_buster}" if os.path.exists(agent_path) else None
        cast_list.append({"name": npc['name'], "appearance": npc['appearance'], "image_url": image_url})
    
    return jsonify({"cast": cast_list})

@app.route('/game/<session_id>/timeline/remove', methods=['POST'])
def remove_from_timeline(session_id):
    data = request.json
    tick, room = data.get("tick"), data.get("room")
    controller = active_sessions.get(session_id)
    if not controller: return jsonify({"error": "Invalid session"}), 404
    video_logger = controller.logger
    success = video_logger.remove_from_timeline(tick, room)
    return jsonify({"message": f"Removed Tick {tick}"}) if success else jsonify({"error": "Failed"}), 400
        
@app.route('/game/<session_id>/upload_cast_image', methods=['POST'])
def upload_cast_image(session_id):
    if 'image' not in request.files: return jsonify({"error": "No image uploaded"}), 400
    file = request.files['image']
    npc_name = request.form.get("npc_name")
    if not file or not npc_name: return jsonify({"error": "Missing params"}), 400
    agent_dir = os.path.join("Assets", "Global_Agents")
    os.makedirs(agent_dir, exist_ok=True)
    filepath = os.path.join(agent_dir, f"{npc_name}.png")
    try:
        img = Image.open(file.stream)
        img.save(filepath, format="PNG")
        safe_url = filepath.replace("\\", "/")
        return jsonify({"message": f"Updated {npc_name}", "image_url": f"/{safe_url}?t={int(time.time())}"})
    except Exception as e:
        return jsonify({"error": f"Failed: {e}"}), 500

@app.route('/game/<session_id>/timeline/spice', methods=['POST'])
def spice_up_timeline(session_id):
    gemini_key, _ = get_reviewer_keys()
    if gemini_key:
        chat_logger.client = genai.Client(api_key=gemini_key)
        
    controller = active_sessions.get(session_id)
    if not controller: return jsonify({"error": "Invalid session"}), 404
    video_logger = controller.logger
    timeline_data = video_logger.get_official_timeline(100) 
    if not timeline_data: return jsonify({"error": "Timeline empty"}), 400

    for scene in timeline_data:
        scene['script'] = [line for line in scene['script'] if line['speaker'].upper() not in ["ACTION", "SYSTEM"]]
    
    result = director.spice_up_story(timeline_data, video_logger)
    return jsonify(result)

@app.route('/game/<session_id>/export_eval_kit', methods=['GET'])
def export_eval_kit(session_id):
    controller = active_sessions.get(session_id)
    if not controller: return jsonify({"error": "Session not found"}), 404
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
    return send_file(memory_file, download_name=f"AIIDE_Eval_Kit_{session_id}.zip", as_attachment=True, mimetype='application/zip')

if __name__ == '__main__':
    app.run(port=5000, debug=True)