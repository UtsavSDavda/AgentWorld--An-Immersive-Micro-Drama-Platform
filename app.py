import os
import uuid
import threading
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from chat_logger import JerichoController, GameDBManager, SQLLogger, AutomatedDirector, SceneSelector
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
    
    # --- WIPE PROTOCOL (Cloud Native) ---
    if clear_db:
        print(f"🗑️ Wiping previous cloud database records for {session_id}...")
        try:
            # Instantiate our new cloud logger temporarily to execute deletes
            logger = SQLLogger(session_id) 
            
            # Fix 1: Change 'messages' to 'chat_logs'
            logger.supabase.table("chat_logs").delete().eq("session_id", session_id).execute()
            
            # Fix 2: Nuke the session_meta to destroy the Z-Machine save state and tick count
            logger.supabase.table("session_meta").delete().eq("session_id", session_id).execute()
            
            logger.supabase.table("room_desc").delete().eq("session_id", session_id).execute()
            logger.supabase.table("official_timeline").delete().eq("session_id", session_id).execute()
            
            # Optional: Wipe character memories/appearances
            # logger.supabase.table("npc_profiles").delete().eq("session_id", session_id).execute()
            
            print("✅ Cloud DB wipe successful!")
        except Exception as e:
            print(f"⚠️ Could not delete cloud DB records: {e}")

    # Spin up the engine state 
    # (Make sure __init__ inside JerichoController passes session_id directly to SupabaseLogger)
    controller = JerichoController(game_path, session_id)
    controller.game_name = safe_filename 
    active_sessions[session_id] = controller
    
    return jsonify({
        "message": f"Welcome back, {username}!" if controller.tick_count > 0 else f"New game started for {username}!", 
        "session_id": session_id,
        "tick": controller.tick_count,
        "game_used": safe_filename
        # Removed "db_name" completely, as we don't return local file paths anymore!
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
    
    video_logger = controller.logger
    
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

    video_logger = controller.logger
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

    video_logger = controller.logger
    
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

    video_logger = controller.logger

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

    video_logger = controller.logger
    
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

    video_logger = controller.logger
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

@app.route('/game/<session_id>/cast', methods=['GET'])
def get_cast(session_id):
    """Fetches the cast list and generates missing initial plates."""
    controller = active_sessions.get(session_id)
    if not controller: return jsonify({"error": "Invalid session"}), 404
    
    # THE FIX: Use the existing, thread-safe logger attached to the controller!
    # No more reverse-engineering the PRAGMA database list.
    npcs = controller.logger.get_all_npcs()
    
    cast_list = []
    for npc in npcs:
        # Use safe paths for cross-platform compatibility
        agent_path = os.path.join("Assets", "Global_Agents", f"{npc['name']}.png")
        
        # If we know the NPC but haven't photographed them yet, do it now!
        if not os.path.exists(agent_path):
            director.create_agent_plate(npc['name'], npc['appearance'])
        
        # Add a timestamp to the URL so the browser never caches an old image
        cache_buster = int(time.time())
        image_url = f"/Assets/Global_Agents/{npc['name']}.png?t={cache_buster}" if os.path.exists(agent_path) else None
        
        cast_list.append({
            "name": npc['name'],
            "appearance": npc['appearance'],
            "image_url": image_url
        })
    
    return jsonify({"cast": cast_list})

@app.route('/game/<session_id>/recast', methods=['POST'])
def recast_character(session_id):
    """Overwrites the character image using a user's custom prompt."""
    data = request.json
    npc_name = data.get("npc_name")
    custom_prompt = data.get("custom_prompt")
    
    if not npc_name or not custom_prompt:
        return jsonify({"error": "Missing name or prompt"}), 400
        
    controller = active_sessions.get(session_id)
    if not controller: return jsonify({"error": "Invalid session"}), 404
    
    video_logger = controller.logger
    
    # 1. Force the director to generate and overwrite the image
    new_path = director.create_agent_plate(
        agent_name=npc_name, 
        agent_desc=custom_prompt, 
        custom_prompt=custom_prompt, 
        force_recreate=True
    )
    
    if new_path:
        # 2. Update the database so future master shots use the new prompt
        video_logger.update_npc_appearance(npc_name, custom_prompt)
        video_logger.close()
        
        safe_url = new_path.replace("\\", "/") # Windows path safety
        return jsonify({
            "message": f"Successfully recast {npc_name}", 
            "image_url": f"/{safe_url}?t={int(time.time())}"
        })
    else:
        video_logger.close()
        return jsonify({"error": "Failed to generate new image"}), 500

@app.route('/game/<session_id>/timeline/remove', methods=['POST'])
def remove_from_timeline(session_id):
    data = request.json
    tick, room = data.get("tick"), data.get("room")
    
    controller = active_sessions.get(session_id)
    if not controller: return jsonify({"error": "Invalid session"}), 404

    video_logger = controller.logger
    success = video_logger.remove_from_timeline(tick, room)
    
    if success:
        return jsonify({"message": f"Removed Tick {tick} from Official Timeline!"})
    else:
        return jsonify({"error": "Failed to remove"}), 400
        
@app.route('/game/<session_id>/upload_cast_image', methods=['POST'])
def upload_cast_image(session_id):
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
        
    file = request.files['image']
    npc_name = request.form.get("npc_name")
    
    if not file or not npc_name:
        return jsonify({"error": "Missing file or character name"}), 400

    # Ensure the directory exists
    agent_dir = os.path.join("Assets", "Global_Agents")
    os.makedirs(agent_dir, exist_ok=True)
    
    filepath = os.path.join(agent_dir, f"{npc_name}.png")
    
    try:
        # Use PIL to ensure it's saved strictly as a PNG, regardless of what the user uploads
        img = Image.open(file.stream)
        img.save(filepath, format="PNG")
        
        safe_url = filepath.replace("\\", "/") # Windows safety
        return jsonify({
            "message": f"Successfully updated {npc_name}", 
            "image_url": f"/{safe_url}?t={int(time.time())}"
        })
    except Exception as e:
        return jsonify({"error": f"Failed to process image: {e}"}), 500
        
if __name__ == '__main__':
    app.run(port=5000, debug=True)