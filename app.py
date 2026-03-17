import os
import uuid
import threading
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS # Required for frontend testing
from chat_logger import JerichoController, GameDBManager, SQLLogger, AutomatedDirector, SceneSelector

OUTPUT_DIR = "Output_Videos"
if not os.path.exists(OUTPUT_DIR): 
    os.makedirs(OUTPUT_DIR)
if not os.path.exists("Assets"): 
    os.makedirs("Assets")

app = Flask(__name__)
CORS(app) # Allow our local HTML file to hit the API

active_sessions = {}
db_manager = GameDBManager()
director = AutomatedDirector()

GAMES_DIR = "games"
if not os.path.exists(GAMES_DIR): os.makedirs(GAMES_DIR)
db_manager.sync_games_directory(GAMES_DIR)

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
    clear_db = data.get("clear_db", False)  # Check if the user wants to wipe the DB
    
    if not game_filename:
         return jsonify({"error": "No game_file provided in request"}), 400
    
    safe_filename = os.path.basename(game_filename)
    game_path = os.path.join(GAMES_DIR, safe_filename)
    
    if not os.path.exists(game_path):
        return jsonify({"error": f"Game file '{safe_filename}' not found."}), 404

    # 1. Get the assigned database path
    db_name = db_manager.get_or_create_db(safe_filename)
    
    # 2. WIPE THE DATABASE IF REQUESTED
    if clear_db and os.path.exists(db_name):
        try:
            os.remove(db_name)
            print(f"🗑️ Wiped previous database for {safe_filename}")
        except Exception as e:
            print(f"⚠️ Could not delete DB file (it might be in use): {e}")

    session_id = str(uuid.uuid4())
    
    # 3. Initialize the controller (If we deleted the DB, SQLLogger will instantly recreate a clean one here)
    controller = JerichoController(game_path, db_name)
    controller.game_name = safe_filename 
    active_sessions[session_id] = controller
    
    return jsonify({
        "message": "Game started", 
        "session_id": session_id,
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
    
    top_scenes = videomaker.scan_and_rank(start_tick=0, end_tick=controller.tick_count, top_n=3)
    
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
        emotion = videomaker._detect_emotion_nrc(line["line"]) 
        formatted_script.append((line["speaker"], line["line"], emotion))
        if line["speaker"] not in speakers:
            assigned_wall = directions[len(speakers) % 4]
            agents.append({"name": line["speaker"], "desc": f"A character", "facing": assigned_wall})
            speakers.add(line["speaker"])

    # Define the exact filename so the frontend knows what to look for
    safe_room = room_name.replace(" ", "_").replace("'", "")
    filename = f"final_render_{controller.game_name}_{safe_room}_tick{tick}.mp4"
    output_path = os.path.join(OUTPUT_DIR, filename)

    def run_film_job():
        scene_assets = director.prepare_scene_assets(scene_data["visual"], agents, controller.game_name, room_name)
        # Pass the full output path (including the Output_Videos folder) to the director
        director.film_scene(formatted_script, scene_assets, controller.game_name, room_name, output_path)
        video_logger.close()
        print(f"✅ Final video saved to {output_path}")

    threading.Thread(target=run_film_job).start()
    
    # Return the expected filename to the frontend
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
    
if __name__ == '__main__':
    app.run(port=5000, debug=True)