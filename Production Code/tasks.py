import os
import subprocess
import json
from huey import SqliteHuey
from google import genai

# Import module reference for dynamic monkey-patching
import chat_logger 
from chat_logger import AutomatedDirector, SQLLogger, SceneSelector

huey = SqliteHuey(filename='render_queue.db')
director = AutomatedDirector()

def apply_reviewer_keys(gemini_key=None, gcp_key=None):
    """Dynamically patches reviewer API keys into the worker process."""
    if gemini_key:
        print("🔑 [WORKER] Applying Reviewer's Gemini Key to worker process...")
        chat_logger.client = genai.Client(api_key=gemini_key)
        chat_logger.video_client = genai.Client(api_key=gemini_key)
        chat_logger.GEMINI_API_KEY = gemini_key
    if gcp_key:
        print("🔑 [WORKER] Applying Reviewer's GCP Key to worker environment...")
        os.environ["GCP_API_KEY"] = gcp_key

@huey.task()
def async_film_scene(scene_visual, agents, formatted_script, game_name, room_name, output_path, mode, gemini_key=None, gcp_key=None):
    apply_reviewer_keys(gemini_key, gcp_key)
    
    print(f"\n🎬 [WORKER] Preparing assets for {room_name}...")
    scene_assets = director.prepare_scene_assets(scene_visual, agents, game_name, room_name)
    
    print(f"🎬 [WORKER] Filming scene (Mode: {mode})...")
    if mode == "stills":
        director.film_scene_stills(formatted_script, scene_assets, game_name, room_name, output_path)
    else:
        director.film_scene(formatted_script, scene_assets, game_name, room_name, output_path)
    print(f"✅ [WORKER] Render complete: {output_path}")

@huey.task()
def async_recap_job(session_id, game_name, past_n, output_path, gemini_key=None, gcp_key=None):
    apply_reviewer_keys(gemini_key, gcp_key)
    
    print(f"\n🎬 [WORKER] Generating recap for {session_id}...")
    video_logger = SQLLogger(session_id)
    director.generate_recap_video(video_logger, game_name, past_n, output_path)
    video_logger.close()
    print(f"✅ [WORKER] Recap complete: {output_path}")

@huey.task()
def async_episode_job(session_id, game_name, timeline_data, output_path, mode, gemini_key=None, gcp_key=None):
    apply_reviewer_keys(gemini_key, gcp_key)
    
    print(f"\n🎬 [WORKER] Compiling Episode for {session_id}...")
    clip_paths = []
    directions = ["north", "east", "south", "west"]
    
    for scene in timeline_data:
        tick = scene["tick"]
        room_name = scene["room"]
        safe_room = room_name.replace(" ", "_").replace("'", "")
        clip_prefix = "animatic" if mode == "stills" else "final_render"
        expected_clip = os.path.join("Output_Videos", f"{clip_prefix}_{game_name}_{safe_room}_tick{tick}.mp4")
        
        if os.path.exists(expected_clip):
            clip_paths.append(expected_clip)
            continue
            
        agents, formatted_script, speakers = [], [], set()
        video_logger = SQLLogger(session_id)
        videomaker = SceneSelector(db=video_logger, director=director, key_terms=[], game_name=game_name)
        
        for line in scene["script"]:
            speaker = line["speaker"]
            emotion = videomaker._detect_emotion_nrc(line["line"])
            voice_id = video_logger.get_npc_voice(speaker)
            formatted_script.append((speaker, line["line"], emotion, voice_id))
            
            if speaker not in speakers:
                assigned_wall = directions[len(speakers) % 4]
                agents.append({"name": speaker, "desc": "A character", "facing": assigned_wall})
                speakers.add(speaker)
        video_logger.close()
        
        scene_assets = director.prepare_scene_assets(scene["visual"], agents, game_name, room_name)
        if mode == "stills":
            director.film_scene_stills(formatted_script, scene_assets, game_name, room_name, expected_clip)
        else:
            director.film_scene(formatted_script, scene_assets, game_name, room_name, expected_clip)
            
        if os.path.exists(expected_clip):
            clip_paths.append(expected_clip)

    if clip_paths:
        list_file = f"ffmpeg_episode_list_{game_name}.txt"
        with open(list_file, "w") as f:
            for clip in clip_paths:
                f.write(f"file '{os.path.abspath(clip)}'\n")
        
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", list_file, "-c", "copy", output_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.remove(list_file)
        print(f"🍿 [WORKER] EPISODE COMPLETE: {output_path}")

@huey.task()
def async_recast_character(session_id, npc_name, custom_prompt, gemini_key=None, gcp_key=None):
    apply_reviewer_keys(gemini_key, gcp_key)
    
    print(f"\n📸 [WORKER] Photographing new plate for {npc_name}...")
    video_logger = SQLLogger(session_id)
    
    new_path = director.create_agent_plate(
        agent_name=npc_name, 
        agent_desc=custom_prompt, 
        custom_prompt=custom_prompt, 
        force_recreate=True
    )
    
    if new_path:
        video_logger.update_npc_appearance(npc_name, custom_prompt)
        print(f"✅ [WORKER] Recast complete for {npc_name}: {new_path}")
    video_logger.close()

@huey.task()
def async_prepare_candidates_assets(session_id, game_name, top_scenes_data, gemini_key=None, gcp_key=None):
    apply_reviewer_keys(gemini_key, gcp_key)
    
    print(f"\n🖼️ [WORKER] Batch compositing pre-viz candidate scenes for {session_id}...")
    video_logger = SQLLogger(session_id)
    directions = ["north", "east", "south", "west"]

    for scene in top_scenes_data:
        room = scene["room"]
        visual_desc = scene["visual"]
        agents = []
        speakers = set()
        
        for line in scene["script"]:
            if line["speaker"] not in speakers:
                assigned_wall = directions[len(speakers) % 4]
                profile = video_logger.get_npc_profile(line["speaker"])
                appearance = profile[1] if profile else f"A character named {line['speaker']}"
                agents.append({
                    "name": line["speaker"], 
                    "desc": appearance, 
                    "facing": assigned_wall
                })
                speakers.add(line["speaker"])
                
        director.prepare_scene_assets(visual_desc, agents, game_name, room)

    video_logger.close()
    print(f"✅ [WORKER] All candidate assets pre-composited and cached for {session_id}!")