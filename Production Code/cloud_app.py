import os
import io
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import anvil.server
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# Replace with the Uplink key generated in your Anvil account
ANVIL_UPLINK_KEY = os.getenv("ANVIL_UPLINK_KEY", "YOUR_UPLINK_KEY_HERE")
print(ANVIL_UPLINK_KEY)
anvil.server.connect(ANVIL_UPLINK_KEY)

def get_reviewer_keys():
    """Extracts custom reviewer keys from HTTP headers."""
    return request.headers.get('X-Gemini-Key'), request.headers.get('X-GCP-Key')

@app.route('/')
def serve_dashboard():
    # Make sure your index_3.html is uploaded to the cloud server folder
    return send_from_directory('.', 'index.html')

@app.route('/games', methods=['GET'])
def list_games():
    return jsonify(anvil.server.call('local_list_games'))

@app.route('/game/start', methods=['POST'])
def start_game():
    data = request.json
    gemini_key, _ = get_reviewer_keys()
    
    # Forward the request to your local PC
    res = anvil.server.call(
        'local_start_game', 
        data.get("game_file"), 
        data.get("username"), 
        data.get("clear_db"), 
        gemini_key
    )
    return jsonify(res)

@app.route('/game/<session_id>/step', methods=['POST'])
def step_game(session_id):
    gemini_key, _ = get_reviewer_keys()
    res = anvil.server.call('local_step_game', session_id, gemini_key)
    if "error" in res: return jsonify(res), 404
    return jsonify(res)

@app.route('/game/<session_id>/orchestrate', methods=['POST'])
def orchestrate_narrative(session_id):
    gemini_key, _ = get_reviewer_keys()
    data = request.json
    res = anvil.server.call('local_orchestrate', session_id, data.get("seed"), gemini_key)
    return jsonify(res)

@app.route('/game/<session_id>/candidates', methods=['POST'])
def get_candidates(session_id):
    gemini_key, gcp_key = get_reviewer_keys()
    res = anvil.server.call('local_get_candidates', session_id, gemini_key, gcp_key)
    return jsonify(res)

@app.route('/game/<session_id>/render', methods=['POST'])
def render_video(session_id):
    gemini_key, gcp_key = get_reviewer_keys()
    data = request.json
    res = anvil.server.call(
        'local_queue_render', 
        session_id, data.get("tick"), data.get("room"), data.get("mode"), 
        gemini_key, gcp_key
    )
    if "error" in res: return jsonify(res), 400
    return jsonify(res)

@app.route('/game/<session_id>/timeline/add', methods=['POST'])
def add_to_timeline(session_id):
    data = request.json
    res = anvil.server.call('local_add_timeline', session_id, data.get("tick"), data.get("room"))
    if "error" in res: return jsonify(res), 400
    return jsonify(res)

@app.route('/game/<session_id>/recap', methods=['POST'])
def create_recap(session_id):
    gemini_key, gcp_key = get_reviewer_keys()
    data = request.json
    res = anvil.server.call('local_queue_recap', session_id, data.get("past_n", 3), gemini_key, gcp_key)
    return jsonify(res)

@app.route('/game/<session_id>/timeline', methods=['GET'])
def get_timeline(session_id):
    res = anvil.server.call('local_get_timeline', session_id)
    return jsonify(res)

@app.route('/game/<session_id>/episode/render', methods=['POST'])
def render_full_episode(session_id):
    gemini_key, gcp_key = get_reviewer_keys()
    data = request.json or {}
    res = anvil.server.call('local_queue_episode', session_id, data.get("mode", "full"), gemini_key, gcp_key)
    if "error" in res: return jsonify(res), 400
    return jsonify(res)

@app.route('/game/<session_id>/cast', methods=['GET'])
def get_cast(session_id):
    res = anvil.server.call('local_get_cast', session_id)
    return jsonify(res)

@app.route('/game/<session_id>/recast', methods=['POST'])
def recast_character(session_id):
    gemini_key, gcp_key = get_reviewer_keys()
    data = request.json
    res = anvil.server.call('local_queue_recast', session_id, data.get("npc_name"), data.get("custom_prompt"), gemini_key, gcp_key)
    if "error" in res: return jsonify(res), 400
    return jsonify(res)

@app.route('/game/<session_id>/timeline/remove', methods=['POST'])
def remove_from_timeline(session_id):
    data = request.json
    res = anvil.server.call('local_remove_timeline', session_id, data.get("tick"), data.get("room"))
    if "error" in res: return jsonify(res), 400
    return jsonify(res)

@app.route('/game/<session_id>/timeline/spice', methods=['POST'])
def spice_up_timeline(session_id):
    gemini_key, _ = get_reviewer_keys()
    res = anvil.server.call('local_spice_timeline', session_id, gemini_key)
    if "error" in res: return jsonify(res), 400
    return jsonify(res)

# ==========================================
# FILE STREAMING OVER ANVIL
# ==========================================

@app.route('/game/<session_id>/upload_cast_image', methods=['POST'])
def upload_cast_image(session_id):
    if 'image' not in request.files: return jsonify({"error": "No image uploaded"}), 400
    file = request.files['image']
    npc_name = request.form.get("npc_name")
    
    # Read the file bytes and package it into an Anvil Media Object
    file_bytes = file.read()
    media_obj = anvil.BlobMedia('image/png', file_bytes, name=file.filename)
    
    res = anvil.server.call('local_upload_cast_image', session_id, npc_name, media_obj)
    if "error" in res: return jsonify(res), 500
    return jsonify(res)

@app.route('/Assets/<path:filename>')
def serve_assets(filename):
    """Pulls the image from the local PC over the Anvil tunnel."""
    media_obj = anvil.server.call('local_fetch_asset', filename)
    if not media_obj: return "File not found", 404
    return send_file(io.BytesIO(media_obj.get_bytes()), mimetype=media_obj.content_type)

@app.route('/videos/<path:filename>')
def serve_videos(filename):
    """Pulls the video from the local PC over the Anvil tunnel."""
    media_obj = anvil.server.call('local_fetch_video', filename)
    if not media_obj: return "File not found", 404
    return send_file(io.BytesIO(media_obj.get_bytes()), mimetype=media_obj.content_type)

@app.route('/render_status/<filename>', methods=['GET'])
def render_status(filename):
    return jsonify(anvil.server.call('local_render_status', filename))

@app.route('/game/<session_id>/export_eval_kit', methods=['GET'])
def export_eval_kit(session_id):
    """Pulls the ZIP file generated by the PC over the tunnel."""
    media_obj = anvil.server.call('local_export_eval_kit', session_id)
    if not media_obj: return jsonify({"error": "Session not found"}), 404
    return send_file(
        io.BytesIO(media_obj.get_bytes()), 
        download_name=f"AIIDE_Eval_Kit_{session_id}.zip", 
        as_attachment=True, 
        mimetype='application/zip'
    )

if __name__ == '__main__':
    # Cloud servers usually bind to 0.0.0.0
    app.run(port=5000, host='0.0.0.0', debug=True)