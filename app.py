import os
import uuid
import threading
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from chat_logger import JerichoController, GameDBManager, SQLLogger, AutomatedDirector, SceneSelector

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

    # The Identity Layer: Bind user to game
    safe_user = "".join(x for x in username if x.isalnum() or x in "-_")
    session_id = f"{safe_user}_{os.path.splitext(safe_filename)[0]}"
    
    db_name = db_manager.get_or_create_user_db(safe_filename, session_id)
    
    # WIPE PROTOCOL
    if clear_db:
        if os.path.exists(db_name):
            try:
                os.remove(db_name)
            except Exception as e:
                print(f"⚠️ Could not delete DB file: {e}")
                
        save_path = os.path.join("Saves", f"{session_id}.sav")
        if os.path.exists(save_path):
            try:
                os.remove(save_path)
                print(f"🗑️ Wiped previous save for {session_id}")
            except Exception as e:
                print(f"⚠️ Could not delete save file: {e}")

    # Re-initialize the bound DB if we just wiped it
    db_name = db_manager.get_or_create_user_db(safe_filename, session_id)
    
    # Spin up the engine state
    controller = JerichoController(game_path, session_id, db_name)
    controller.game_name = safe_filename 
    active_sessions[session_id] = controller
    
    return jsonify({
        "message": f"Welcome back, {username}!" if controller.tick_count > 0 else f"New game started for {username}!", 
        "session_id": session_id,
        "tick": controller.tick_count,
        "db_name": db_name,
        "game_used": safe_filename
    })

@app.route('/game/<session_id>/step', methods=['POST'])
def step_game(session_id):
    controller = active_sessions.get(session_id)
    if not controller: return jsonify({"error": "Invalid session"}), 404
    locations = controller.step()
    return jsonify({"tick": controller.tick_count, "locations": locations})

@app.route('/game/<session_id>/candidates', methods=['POST'])
def get_candidates(session_id):
    controller = active_sessions.get(session_id)
    if not controller: return jsonify({"error": "Invalid session"}), 404
    
    db_name = controller.logger.conn.cursor().connection.execute("PRAGMA database_list").fetchall()[0][2] 
    video_logger = SQLLogger(db_name)
    
    game_terms = ["darkness", "grue", "lamp", "spell"]
    videomaker = SceneSelector(db=video_logger, director=director, key_terms=game_terms, game_name=controller.game_name)
    
    top_scenes = videomaker.scan_and_rank(start_tick=0, end_tick=controller.tick_count)
    
    candidates = []
    directions = ["north", "east", "south", "west"]

    for score, tick, room, scene_data in top_scenes:
        agents = []
        speakers = set()
        for line in scene_data["script"]:
            if line["speaker"] not in speakers:
                # Assign the next wall based on how many speakers we already have
                assigned_wall = directions[len(speakers) % 4]
                profile = video_logger.get_npc_profile(line["speaker"])
                print(profile)
                appearance = profile[1] if profile else f"A character named {line['speaker']}"
                agents.append({
                    "name": line["speaker"], 
                    "desc": appearance,  # Passes the exact visual prompt!
                    "facing": assigned_wall
                })
                speakers.add(line["speaker"])
        # This calls your new MediaPipe pipeline!
        scene_assets = director.prepare_scene_assets(scene_data["visual"], agents, controller.game_name, room)
        
        candidates.append({
            "tick": tick,
            "room": room,
            "score": score,
            "script": scene_data["script"],
            "images": scene_assets
        })
        
    return jsonify({"candidates": candidates})

@app.route('/videos/<path:filename>')
def serve_videos(filename):
    return send_from_directory(OUTPUT_DIR, filename)

@app.route('/render_status/<filename>', methods=['GET'])
def render_status(filename):
    # If the file exists in the folder, the ffmpeg stitch is complete!
    file_path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(file_path):
        return jsonify({"status": "done", "video_url": f"/videos/{filename}"})
    else:
        return jsonify({"status": "processing"})

@app.route('/game/<session_id>/render', methods=['POST'])
def render_video(session_id):
    data = request.json
    tick, room_name = data.get("tick"), data.get("room")
    mode = data.get("mode", "full") # Added mode parameter
    
    controller = active_sessions.get(session_id)
    if not controller: return jsonify({"error": "Invalid session"}), 404

    db_name = controller.logger.conn.cursor().connection.execute("PRAGMA database_list").fetchall()[0][2]
    video_logger = SQLLogger(db_name)
    scene_data = video_logger.get_structured_scene_data(tick, room_name)
    
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
    
    # Change filename prefix based on the mode so they don't overwrite each other
    prefix = "animatic" if mode == "stills" else "final_render"
    filename = f"{prefix}_{controller.game_name}_{safe_room}_tick{tick}.mp4"
    output_path = os.path.join(OUTPUT_DIR, filename)

    def run_film_job():
        scene_assets = director.prepare_scene_assets(scene_data["visual"], agents, controller.game_name, room_name)
        
        # Branch logic based on user selection
        if mode == "stills":
            director.film_scene_stills(formatted_script, scene_assets, controller.game_name, room_name, output_path)
        else:
            director.film_scene(formatted_script, scene_assets, controller.game_name, room_name, output_path)
            
        video_logger.close()
        print(f"✅ Video saved to {output_path}")

    threading.Thread(target=run_film_job).start()
    
    return jsonify({
        "message": f"Render started for Tick {tick}!", 
        "video_filename": filename 
    })

@app.route('/game/<session_id>/timeline/add', methods=['POST'])
def add_to_timeline(session_id):
    data = request.json
    tick, room = data.get("tick"), data.get("room")
    
    controller = active_sessions.get(session_id)
    if not controller: return jsonify({"error": "Invalid session"}), 404

    db_name = controller.logger.conn.cursor().connection.execute("PRAGMA database_list").fetchall()[0][2]
    video_logger = SQLLogger(db_name)
    
    success = video_logger.add_to_timeline(tick, room)
    video_logger.close()
    
    if success:
        return jsonify({"message": f"Added Tick {tick} to Official Timeline!"})
    else:
        return jsonify({"error": "Already in timeline"}), 400

@app.route('/game/<session_id>/recap', methods=['POST'])
def create_recap(session_id):
    data = request.json
    past_n = data.get("past_n", 3)
    
    controller = active_sessions.get(session_id)
    if not controller: return jsonify({"error": "Invalid session"}), 404

    db_name = controller.logger.conn.cursor().connection.execute("PRAGMA database_list").fetchall()[0][2]
    video_logger = SQLLogger(db_name)

    filename = f"Recap_{controller.game_name}_{uuid.uuid4().hex[:6]}.mp4"
    output_path = os.path.join(OUTPUT_DIR, filename)

    def run_recap_job():
        director.generate_recap_video(video_logger, controller.game_name, past_n, output_path)
        video_logger.close()

    import threading
    threading.Thread(target=run_recap_job).start()
    
    return jsonify({
        "message": f"Generating recap of last {past_n} events...",
        "video_filename": filename
    })

@app.route('/game/<session_id>/timeline', methods=['GET'])
def get_timeline(session_id):
    """Returns the current official timeline for the Storyboard UI."""
    controller = active_sessions.get(session_id)
    if not controller: return jsonify({"error": "Invalid session"}), 404

    db_name = controller.logger.conn.cursor().connection.execute("PRAGMA database_list").fetchall()[0][2]
    video_logger = SQLLogger(db_name)
    
    # Fetch up to the last 100 pinned scenes (chronological)
    timeline_data = video_logger.get_official_timeline(100) 
    video_logger.close()
    
    return jsonify({"timeline": timeline_data})

@app.route('/game/<session_id>/episode/render', methods=['POST'])
def render_full_episode(session_id):
    """Iterates through the timeline, renders missing clips (Animatic or Veo), and stitches them."""
    data = request.json or {}
    mode = data.get("mode", "full") # Default to full Veo if not specified
    
    controller = active_sessions.get(session_id)
    if not controller: return jsonify({"error": "Invalid session"}), 404

    db_name = controller.logger.conn.cursor().connection.execute("PRAGMA database_list").fetchall()[0][2]
    video_logger = SQLLogger(db_name)
    timeline_data = video_logger.get_official_timeline(100)
    
    if not timeline_data:
        return jsonify({"error": "Timeline is empty. Pin some scenes first!"}), 400

    # Differentiate the final output filename based on the mode
    prefix = "Animatic_Episode" if mode == "stills" else "Episode_Full"
    episode_filename = f"{prefix}_{controller.game_name}_{uuid.uuid4().hex[:6]}.mp4"
    output_path = os.path.join(OUTPUT_DIR, episode_filename)

    def run_episode_job():
        clip_paths = []
        directions = ["north", "east", "south", "west"]
        
        mode_label = "ANIMATIC" if mode == "stills" else "VEO"
        print(f"\n🎬 --- EPISODE COMPILER STARTED ({mode_label} MODE) ---")
        
        for scene in timeline_data:
            tick = scene["tick"]
            room_name = scene["room"]
            safe_room = room_name.replace(" ", "_").replace("'", "")
            
            # Look for the correct clip type based on the requested mode
            clip_prefix = "animatic" if mode == "stills" else "final_render"
            expected_clip = os.path.join(OUTPUT_DIR, f"{clip_prefix}_{controller.game_name}_{safe_room}_tick{tick}.mp4")
            
            if os.path.exists(expected_clip):
                print(f"♻️ Found existing {mode_label} render for Tick {tick}. Skipping generation.")
                clip_paths.append(expected_clip)
            else:
                print(f"🎥 Rendering missing {mode_label} clip for Tick {tick}...")
                
                agents, formatted_script, speakers = [], [], set()
                videomaker = SceneSelector(db=video_logger, director=director, key_terms=[], game_name=controller.game_name)
                
                for line in scene["script"]:
                    speaker = line["speaker"]
                    emotion = videomaker._detect_emotion_nrc(line["line"])
                    voice_id = video_logger.get_npc_voice(speaker)
                    formatted_script.append((speaker, line["line"], emotion, voice_id))
                    
                    if speaker not in speakers:
                        assigned_wall = directions[len(speakers) % 4]
                        agents.append({"name": speaker, "desc": "A character", "facing": assigned_wall})
                        speakers.add(speaker)
                
                scene_assets = director.prepare_scene_assets(scene["visual"], agents, controller.game_name, room_name)
                
                # Route to the correct filming method
                if mode == "stills":
                    director.film_scene_stills(formatted_script, scene_assets, controller.game_name, room_name, expected_clip)
                else:
                    director.film_scene(formatted_script, scene_assets, controller.game_name, room_name, expected_clip)
                
                if os.path.exists(expected_clip):
                    clip_paths.append(expected_clip)

        if clip_paths:
            print(f"🎞️ Stitching {len(clip_paths)} scenes into final episode...")
            list_file = f"ffmpeg_episode_list_{controller.game_name}.txt"
            with open(list_file, "w") as f:
                for clip in clip_paths:
                    f.write(f"file '{os.path.abspath(clip)}'\n")
            
            import subprocess
            subprocess.run([
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", list_file, "-c", "copy", output_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            os.remove(list_file)
            print(f"🍿 EPISODE COMPLETE: {output_path}")
            
        video_logger.close()

    import threading
    threading.Thread(target=run_episode_job).start()
    
    return jsonify({
        "message": f"{prefix.replace('_', ' ')} compilation started!", 
        "video_filename": episode_filename 
    })

if __name__ == '__main__':
    app.run(port=5000, debug=True)