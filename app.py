from flask import Flask, request, jsonify
import uuid
import os
import threading

# Import your existing classes
from chat_logger import JerichoController, GameDBManager, SQLLogger, AutomatedDirector, SceneSelector

app = Flask(__name__)

# Global dictionary to hold active game instances in memory (for v1 prototype)
active_sessions = {}
db_manager = GameDBManager()
director = AutomatedDirector()


#Functions
def generate_game_terms_via_llm(game_name, initial_context):
    # Your LLM call here...
    # Returns: ["alien", "laser", "airlock", "oxygen"]
    pass

@app.route('/game/start', methods=['POST'])
def start_game():
    data = request.json
    game_file = data.get("game_file", "Control4.z8")
    
    if not os.path.exists(game_file):
        return jsonify({"error": "Game file not found"}), 404

    session_id = str(uuid.uuid4())
    db_name = db_manager.get_or_create_db(game_file)
    
    # Initialize the controller and store it in our global sessions
    controller = JerichoController(game_file, db_name)
    active_sessions[session_id] = controller
    
    return jsonify({
        "message": "Game started", 
        "session_id": session_id,
        "db_name": db_name
    })

@app.route('/game/<session_id>/step', methods=['POST'])
def step_game(session_id):
    if session_id not in active_sessions:
        return jsonify({"error": "Invalid or expired session"}), 404
        
    controller = active_sessions[session_id]
    
    # Run ONE tick of the simulation instead of a while loop
    locations = controller.step()
    
    return jsonify({
        "tick": controller.tick_count,
        "locations": locations
    })

@app.route('/game/<session_id>/generate_video', methods=['POST'])
def generate_video(session_id):
    data = request.json
    start_tick = data.get("start_tick", 0)
    end_tick = data.get("end_tick", 5)
    
    if session_id not in active_sessions:
        return jsonify({"error": "Invalid session"}), 404
        
    controller = active_sessions[session_id]
    db_name = controller.logger.conn.cursor().connection.execute("PRAGMA database_list").fetchall()[0][2] # Hack to get current DB path
    # WARNING: Video generation takes minutes. In a real API, you cannot do this synchronously.
    # For a quick prototype, we wrap it in a background thread.
    def run_video_gen():
        video_logger = SQLLogger(db_name)
        videomaker = SceneSelector(db=video_logger, director=director, key_terms=game_terms)
        game_terms = generate_game_terms_via_llm(game_name, initial_context)
        # Note: You'll need to modify 'run_auto' to not ask for input(), but to accept an index automatically
        videomaker.run_auto(start_tick=start_tick, end_tick=end_tick) 
        video_logger.close()

    thread = threading.Thread(target=run_video_gen)
    thread.start()
    
    return jsonify({"message": "Video generation started in the background. Check server logs."})

if __name__ == '__main__':
    # Run on all available IPs, port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)