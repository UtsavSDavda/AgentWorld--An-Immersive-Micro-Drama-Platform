import jericho
import re
import sqlite3
import time
import os
from google import genai
from typing import Dict, List
from dotenv import load_dotenv
from videoprompts import SYSTEM_PROMPT,EPISODE
import random
from google.genai import types
import requests
import subprocess
import heapq
import statistics
from textblob import TextBlob
import json
from nrclex import NRCLex
import csv 
from PIL import Image
import io
import base64
import cv2
import numpy as np
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pickle
from supabase import create_client, Client
import mimetypes

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
print(GEMINI_API_KEY)
client = genai.Client(api_key=GEMINI_API_KEY)
model = "gemini-2.5-flash"
vid_model = "veo-3.1-fast-generate-preview"
IMG_MODEL = "imagen-4.0-fast-generate-001"
VID_MODEL = "veo-3.1-fast-generate-preview"
TEXT_MODEL = "gemini-2.5-flash"
video_client = genai.Client(api_key=GEMINI_API_KEY)

# --- DATABASE LOGGER ---
def split_into_sentences(text):
    """Splits a paragraph into a list of individual sentences."""
    if not text:
        return []
    # Splits by ., !, or ? followed by a space and a capital letter, or end of string.
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$', text.strip())
    return [s.strip() for s in sentences if s.strip()]

def create_video(tick,context,prompt):
    print("Generating video...")
    no = random.randint(1,100)
    prompt.format(context=context)
    print(prompt)
    operation = video_client.models.generate_videos(
        model=vid_model,
        prompt=prompt.format(context=context),
        config=types.GenerateVideosConfig(
            number_of_videos=1,
            aspect_ratio="16:9"
        )
    )

    while not operation.done:
        print("Waiting for video to finish processing...")
        time.sleep(5)
        operation = client.operations.get(operation)

    if operation.result:
        generated_video = operation.result.generated_videos[0]
        print("Video generated successfully!")
        client.files.download(file=generated_video.video)
        name = f"tick{tick}_{no}.mp4"
        generated_video.video.save(name)
        print(f"Saved to {name}")
    else:
        print("Video generation failed.")

def produce_video_from_tick(tick, room_name, logger, director):
    """
    Orchestrates the MovieDirector to film a specific moment in history.
    """
    print(f"\n🎬 --- PRODUCTION STARTED: {room_name} (Tick {tick}) ---")
    
    # 1. GET DATA FROM DB
    scene_data = logger.get_structured_scene_data(tick, room_name)
    
    if not scene_data["script"]:
        print("❌ No dialogue found for this tick. Production halted.")
        return

    visual_context = scene_data["visual"]
    script = scene_data["script"]
    
    print(f"📍 Set Description: {visual_context}")
    print(f"📝 Script Length: {len(script)} lines")

    # 2. CASTING (Generate Master Shots for actors present)
    # We find all unique speakers in this scene
    present_actors = set(line["speaker"] for line in script)
    print(script)
    for actor in present_actors:
        # Check if Director already has this asset, if not, generate it
        if actor not in director.assets:
            desc = NPC_APPEARANCE.get(actor, NPC_APPEARANCE["DEFAULT"])
            
            # Generate the master shot
            director.generate_master_shot(
                agent_name=actor,
                agent_desc=desc,
                room_desc=visual_context, # Use the actual room description from DB!
                facing="right" # Defaulting to right for simplicity
            )

    # 3. FILMING (Generate Video Clips)
    clip_filenames = []
    
    for shot_idx, entry in enumerate(script):
        speaker = entry["speaker"]
        line = entry["line"]
        
        # Simple heuristic for emotion based on punctuation
        emotion = "neutral"
        if "!" in line: emotion = "excited"
        elif "?" in line: emotion = "inquisitive"
        elif "..." in line: emotion = "hesitant"
        
        print(f"\n🎥 Shot {shot_idx+1}/{len(script)}: {speaker}")
        
        # Call the Director to film the specific line
        clip_path = director.film_scene(
            speaker_name=speaker,
            dialogue_line=line,
            emotion=emotion
        )
        
        if clip_path:
            clip_filenames.append(clip_path)
            
    # 4. POST-PRODUCTION (Stitch)
    if clip_filenames:
        output_file = f"Ep_{tick}_{room_name}.mp4"
        director.stitch_movie(clip_filenames, output_name=output_file)
        print(f"\n🌟 THAT'S A WRAP! Final file: {output_file}")
    else:
        print("❌ Production failed: No valid clips generated.")

def get_media_duration(file_path):
    """Returns the duration of a media file in seconds."""
    try:
        result = subprocess.run([
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration", "-of",
            "default=noprint_wrappers=1:nokey=1", file_path
        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        return float(result.stdout)
    except Exception as e:
        print(f"❌ Failed to get duration for {file_path}: {e}")
        return 0.0


def fetch_gcp_casting_menu(output_file="gcp_voices.json"):
    api_key = os.getenv("GCP_API_KEY", "")
    if not api_key:
        print("❌ Error: GCP_API_KEY not found in environment variables.")
        return
        
    url = f"https://texttospeech.googleapis.com/v1/voices?key={api_key}"
    
    print("📋 Fetching voice catalog from Google Cloud...")
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"❌ Failed to fetch voices: {response.text}")
        return

    all_voices = response.json().get("voices", [])
    
    # We only want premium English voices for cinematic quality
    premium_tiers = ["Journey", "Neural2", "Studio"]
    catalog = {"MALE": [], "FEMALE": [], "NEUTRAL": []}
    
    for v in all_voices:
        name = v["name"]
        gender = v.get("ssmlGender", "NEUTRAL")
        lang = v["languageCodes"][0]
        
        # Filter for English and Premium models
        if "en-" in lang and any(tier in name for tier in premium_tiers):
            catalog[gender].append(name)

    with open(output_file, 'w') as f:
        json.dump(catalog, f, indent=4)
        
    print(f"✅ Saved premium voice catalog to {output_file}")
    print(f"Found {len(catalog['MALE'])} Male, {len(catalog['FEMALE'])} Female, and {len(catalog['NEUTRAL'])} Neutral voices.")

def cast_random_gcp_voice(gender, catalog_path="gcp_voices.json"):
    """Randomly selects a voice from the catalog based on gender."""
    target_gender = "FEMALE" if gender.upper() == "FEMALE" else "MALE"
        
    try:
        with open(catalog_path, 'r') as f:
            catalog = json.load(f)
    except FileNotFoundError:
        return "en-US-Neural2-J" # Ultimate fallback
        
    available_voices = catalog.get(target_gender, [])
    if not available_voices:
        return "en-US-Neural2-J"
        
    return random.choice(available_voices)

class GameDBManager:
    DB_FOLDER = "GamesDB"
    # Put the registry file inside the folder too to keep things clean
    REGISTRY_FILE = os.path.join("GamesDB", "game_registry.csv") 

    def __init__(self):
        # Ensure the folder exists before doing anything
        if not os.path.exists(self.DB_FOLDER):
            os.makedirs(self.DB_FOLDER)

        if not os.path.exists(self.REGISTRY_FILE):
            with open(self.REGISTRY_FILE, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["game_file", "db_name"])

    def get_or_create_user_db(self, game_file, session_id):
        """Creates a strictly isolated database for this specific user and game."""
        game_file = os.path.basename(game_file)
        raw_db_name = f"db_{os.path.splitext(game_file)[0]}_{session_id}.db"
        db_name = os.path.join(self.DB_FOLDER, raw_db_name)

        import sqlite3
        try:
            conn = sqlite3.connect(db_name)
            cursor = conn.cursor()
            # Initialize base schema and session metadata
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS room_desc (
                room_name TEXT,
                description TEXT,
                tick INTEGER,
                PRIMARY KEY (room_name, tick)
            );
            """)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_meta (
                key TEXT PRIMARY KEY,
                value TEXT
            );
            """)
            conn.commit()
            conn.close()
            print(f"🗄️ Bound database to session: {db_name}")
        except sqlite3.Error as e:
            print(f"❌ Failed to initialize database file {db_name}: {e}")

        return db_name

    def sync_games_directory(self, games_dir):
        # We no longer pre-generate DBs here since they are user-dependent.
        # Just ensure the directory exists.
        if not os.path.exists(games_dir):
            os.makedirs(games_dir)
        print("✅ Games directory synced.")

class DramaScorer:
    def __init__(self, key_terms=None):
        # 1. Domain-Specific Terms (Customize for your specific Jericho game)
        # Mentions of these increase the 'Relevance' score.
        self.key_terms = key_terms if key_terms else [
            "gun", "key", "password", "kill", "love", "secret", "hide", 
            "traitor", "escape", "code", "money", "body"
        ]
        
        # 2. Conflict Triggers
        # Words that suggest disagreement, denial, or interruption.
        self.conflict_words = {
            "no", "never", "stop", "lie", "liar", "wrong", "wait", 
            "don't", "cant", "can't", "hate", "die", "dead"
        }

    def _get_sentiment_data(self, dialogues):
        """Returns a list of polarity scores (-1.0 to 1.0) for the sequence."""
        polarities = []
        for line in dialogues:
            blob = TextBlob(line['text'])
            polarities.append(blob.sentiment.polarity)
        return polarities

    def _calculate_pacing_score(self, dialogues):
        """
        Analyzes the rhythm. 
        - High score for 'snappy' back-and-forth (short lines).
        - Penalty for long monologues (boring for video).
        """
        word_counts = [len(line['text'].split()) for line in dialogues]
        avg_len = statistics.mean(word_counts)
        
        # Ideal micro-drama line length is 5-15 words.
        if 5 <= avg_len <= 15:
            return 1.0
        elif avg_len < 5: 
            return 0.7 # Too short
        else:
            return 0.5 # Too verbose

    def _calculate_conflict_score(self, dialogues):
        """Checks for direct conflict keywords and imperative punctuation."""
        score = 0
        total_lines = len(dialogues)
        
        for line in dialogues:
            text_lower = line['text'].lower()
            
            # Check for conflict keywords
            if any(word in text_lower for word in self.conflict_words):
                score += 1.5
            
            # Check for dramatic punctuation (! or ?)
            if "?" in line['text'] or "!" in line['text']:
                score += 1.0

        # Normalize relative to length (prevent long boring logs from winning just by volume)
        normalized_score = min((score / total_lines) * 10, 10) 
        return normalized_score

    def _calculate_volatility(self, polarities):
        """
        Measures emotional instability.
        Standard Deviation of sentiment: High std_dev = High Drama.
        """
        if len(polarities) < 2:
            return 0
        
        # Standard deviation gives us how much the tone bounces around
        volatility = statistics.stdev(polarities)
        
        # Normalize: A std_dev of 0.5 is extremely high for sentiment
        return min(volatility * 20, 10) 

    def calculate_score(self, dialogues):
        """
        Main entry point.
        Input: List of dicts [{'speaker': 'A', 'text': '...'}, ...]
        Output: Final Score (0-100)
        """
        if not dialogues:
            return 0

        # 1. Get raw metrics
        polarities = self._get_sentiment_data(dialogues)
        
        # 2. Calculate Component Scores
        volatility_score = self._calculate_volatility(polarities) # Max 10
        conflict_score = self._calculate_conflict_score(dialogues) # Max 10
        pacing_multiplier = self._calculate_pacing_score(dialogues) # 0.5 - 1.0
        
        # 3. Key Term Bonus (The "Plot Relevance" boost)
        term_matches = 0
        all_text = " ".join([d['text'].lower() for d in dialogues])
        for term in self.key_terms:
            if term in all_text:
                term_matches += 1
        relevance_bonus = min(term_matches * 5, 20) # Max 20 points bonus
        
        # 4. Sentiment Extremes (Did someone get really angry or really happy?)
        extreme_bonus = 0
        if min(polarities) < -0.5 or max(polarities) > 0.6:
            extreme_bonus = 10

        # --- THE FORMULA ---
        # Base Score (max 30) + Conflict (max 30) + Bonuses
        # We weight Conflict and Volatility heavily for drama.
        
        raw_score = (
            (volatility_score * 3.0) +  # Emotional whiplash is #1 for video
            (conflict_score * 3.0) +    # Arguments are interesting
            relevance_bonus +           # Plot relevance
            extreme_bonus               # High stakes moments
        )
        
        # Apply Pacing Multiplier
        final_score = raw_score * pacing_multiplier
        
        return round(min(final_score, 100), 2)

class RoomGeometry:
    """
    Defines the strict physical constraints of the room.
    This prevents 'hallucinating' new rooms for every shot.
    """
    def __init__(self, materials, north_wall, south_wall, east_wall, west_wall, is_indoors=True):
        self.materials = materials  # The "Global Skin" (e.g. rusted metal, neon lights)
        self.walls = {
            "north": north_wall,
            "south": south_wall,
            "east": east_wall,
            "west": west_wall
        }
        self.is_indoors = is_indoors

    def get_background_prompt(self, facing_direction):
        """
        Returns the specific wall description based on where the agent is looking.
        """
        # If Agent faces North, the BACKGROUND is the North Wall.
        # (Technically in film, if I face North, the background is North).
        
        wall_feature = self.walls.get(facing_direction.lower(), self.walls["north"])
        
        prompt = f"""
        SET DESIGN / BACKGROUND:
        Location materials: {self.materials}.
        Visible background feature: {wall_feature}.
        """
        
        if self.is_indoors:
            prompt += " Environment: INTERIOR. Enclosed space. Ceiling visible. No sky."
            
        return prompt
        
class SceneArchitect:
    """
    Ensures visual consistency by 'designing' the room once and enforcing
    that description across all camera angles.
    """
    def __init__(self):
        self.room_anchors = {} # Stores the strict descriptions

    def design_set(self, raw_room_desc):
        """
        Uses an LLM to expand a vague room description into a rigid visual anchor.
        """
        print(f"🏗️  Designing set for: '{raw_room_desc}'...")
        
        # We ask the LLM to be a 'Set Designer'
        prompt = f"""
        Act as a visual set designer for a movie. 
        I will give you a location: "{raw_room_desc}".
        
        Write a concise but extremely specific visual description of the background materials.
        Include:
        1. Specific wall texture/color (e.g., "rusted corrugated iron").
        2. Specific lighting color (e.g., "hazy blue neon").
        3. Specific floor material.
        
        Output ONLY the description string. Do not add intro text.
        Keep it under 40 words.
        """
        
        response = client.models.generate_content(
            model=TEXT_MODEL,
            contents=prompt
        )
        
        anchor_text = response.text.strip()
        self.room_anchors[raw_room_desc] = anchor_text
        print(f"🔒 Set Locked: [{anchor_text}]")
        return anchor_text

class SQLLogger:
    def __init__(self, session_id):
        url: str = os.getenv("SUPABASE_URL")
        key: str = os.getenv("SUPABASE_KEY")
        if not url or not key:
            raise ValueError("Supabase credentials missing in .env")
        
        self.supabase: Client = create_client(url, key)
        self.session_id = session_id

    def save_current_tick(self, tick_count):
        self.supabase.table('session_meta').upsert({
            'session_id': self.session_id,
            'key': 'current_tick',
            'value': str(tick_count)
        }).execute()

    def get_saved_tick(self):
        res = self.supabase.table('session_meta').select('value')\
            .eq('session_id', self.session_id).eq('key', 'current_tick').execute()
        return int(res.data[0]['value']) if res.data else 0

    def save_npc_profile(self, name, raw_desc, persona, appearance, gender="NEUTRAL", voice_id="en-US-Neural2-J"):
        self.supabase.table('npc_profiles').upsert({
            'session_id': self.session_id,
            'name': name,
            'raw_description': raw_desc,
            'persona': persona,
            'appearance': appearance,
            'gender': gender,
            'voice_id': voice_id
        }).execute()

    def get_npc_profile(self, name):
        res = self.supabase.table('npc_profiles').select('persona, appearance')\
            .eq('session_id', self.session_id).eq('name', name).execute()
        return (res.data[0]['persona'], res.data[0]['appearance']) if res.data else None

    def get_npc_voice(self, name):
        res = self.supabase.table('npc_profiles').select('voice_id')\
            .eq('session_id', self.session_id).eq('name', name).execute()
        return res.data[0]['voice_id'] if res.data else "en-US-Neural2-J"

    def get_all_npcs(self):
        res = self.supabase.table('npc_profiles').select('*')\
            .eq('session_id', self.session_id).execute()
        return res.data

    def update_npc_appearance(self, name, new_appearance_prompt):
        self.supabase.table('npc_profiles').update({
            'appearance': new_appearance_prompt
        }).eq('session_id', self.session_id).eq('name', name).execute()

    def update_room_desc(self, room_name, description, tick):
        data = {
            "session_id": self.session_id, # Make sure this matches your class variable name
            "room_name": room_name,
            "description": description,
            "tick": tick
        }
        
        try:
            # The on_conflict parameter is the magic key here!
            self.supabase.table("room_desc").upsert(
                data, 
                on_conflict="session_id,room_name,tick" 
            ).execute()
        except Exception as e:
            print(f"⚠️ Failed to upsert room description: {e}")

    def log_broadcast(self, tick, room, sender, receivers, message):
        """Unified logging: One row captures the whole event."""
        recipients_str = ", ".join(receivers)
        self.supabase.table('chat_logs').insert({
            'session_id': self.session_id,
            'tick': tick,
            'room_name': room,
            'sender': sender,
            'receiver': recipients_str,
            'message': message
        }).execute()

    def get_agent_context(self, agent_name, limit=5):
        """Finds messages sent by the agent, or where the agent is in the receiver list."""
        res = self.supabase.table('chat_logs').select('sender, message, tick')\
            .eq('session_id', self.session_id)\
            .or_(f"sender.eq.{agent_name},receiver.ilike.%{agent_name}%")\
            .order('id', desc=True).limit(limit).execute()
            
        rows = res.data[::-1] # Reverse chronologically
        return "\n".join([f"[Tick {r['tick']}] {r['sender']}: {r['message']}" for r in rows])

    def get_structured_scene_data(self, tick, room_name):
        # 1. Fetch Room Visuals (Closest tick <= current tick)
        res_room = self.supabase.table('room_desc').select('description')\
            .eq('session_id', self.session_id).eq('room_name', room_name)\
            .lte('tick', tick).order('tick', desc=True).limit(1).execute()
            
        visual_desc = res_room.data[0]['description'] if res_room.data else "A dark, empty room."

        # 2. Fetch Dialogue for this exact tick
        res_chat = self.supabase.table('chat_logs').select('sender, message')\
            .eq('session_id', self.session_id).eq('room_name', room_name)\
            .eq('tick', tick).order('id').execute()
        
        script = [{"speaker": row['sender'], "line": row['message']} for row in res_chat.data]
        
        return {
            "visual": visual_desc,
            "script": script,
            "room": room_name,
            "tick": tick
        }

    def get_rooms_for_tick(self, tick):
        """Finds any room that had activity on this tick."""
        res = self.supabase.table('chat_logs').select('room_name')\
            .eq('session_id', self.session_id).eq('tick', tick).execute()
            
        # Extract unique room names
        return list(set([row['room_name'] for row in res.data]))

    def add_to_timeline(self, tick, room_name):
        try:
            self.supabase.table('official_timeline').insert({
                'session_id': self.session_id,
                'tick': tick,
                'room_name': room_name
            }).execute()
            return True
        except Exception as e:
            print(f"⚠️ Scene already pinned: {e}")
            return False

    def get_official_timeline(self, past_n_ticks):
        res = self.supabase.table('official_timeline').select('tick, room_name')\
            .eq('session_id', self.session_id).order('tick', desc=True).limit(past_n_ticks).execute()
            
        rendered_scenes = res.data[::-1] 
        full_history = []
        for scene in rendered_scenes:
            data = self.get_structured_scene_data(scene['tick'], scene['room_name'])
            full_history.append(data)
            
        return full_history
    
    def save_zmachine_state(self, b64_state):
        self.supabase.table('session_meta').upsert({
            'session_id': self.session_id,
            'key': 'zmachine_state',
            'value': b64_state
        }).execute()

    def get_zmachine_state(self):
        res = self.supabase.table('session_meta').select('value')\
            .eq('session_id', self.session_id).eq('key', 'zmachine_state').execute()
        return res.data[0]['value'] if res.data else None
    
    def remove_from_timeline(self, tick, room_name):
        try:
            self.supabase.table('official_timeline').delete().eq('session_id', self.session_id).eq('tick', tick).eq('room_name', room_name).execute()
            return True
        except Exception as e:
            print(f"⚠️ Failed to remove scene from timeline: {e}")
            return False

    def overwrite_scene_dialogue(self, tick, room_name, new_script):
        """Deletes old chat logs for a specific scene and inserts the rewritten AI script."""
        try:
            # 1. Nuke the old, boring conversation
            self.supabase.table('chat_logs').delete().eq('session_id', self.session_id).eq('tick', tick).eq('room_name', room_name).execute()
            
            # 2. Insert the new, dramatic lines
            for msg in new_script:
                self.supabase.table('chat_logs').insert({
                    'session_id': self.session_id,
                    'tick': tick,
                    'room_name': room_name,
                    'sender': msg['speaker'],
                    'receiver': 'ALL', # Simplified for the rewrite
                    'message': msg['line']
                }).execute()
            return True
        except Exception as e:
            print(f"⚠️ Failed to overwrite dialogue for Tick {tick}: {e}")
            return False

    def overwrite_narrative_arc(self, min_tick, max_tick, new_scenes):
        """Wipes a chunk of simulation history and replaces it with the AI's cinematic pacing."""
        try:
            # 1. Clear the old, boring timeline and chat logs for this entire block
            self.supabase.table('official_timeline').delete().eq('session_id', self.session_id).gte('tick', min_tick).lte('tick', max_tick).execute()
            self.supabase.table('chat_logs').delete().eq('session_id', self.session_id).gte('tick', min_tick).lte('tick', max_tick).execute()

            # 2. Insert the newly paced scenes. 
            # We assign them sequential ticks starting from min_tick.
            current_tick = min_tick
            for scene in new_scenes:
                room_name = scene['room']
                
                # Pin the new scene to the official timeline
                self.supabase.table('official_timeline').insert({
                    'session_id': self.session_id,
                    'tick': current_tick,
                    'room_name': room_name
                }).execute()
                
                # Insert the new, dramatic dialogue
                for msg in scene['script']:
                    self.supabase.table('chat_logs').insert({
                        'session_id': self.session_id,
                        'tick': current_tick,
                        'room_name': room_name,
                        'sender': msg['speaker'],
                        'receiver': 'ALL',
                        'message': msg['line']
                    }).execute()
                    
                current_tick += 1 # Advance Narrative Time
                
            return True
        except Exception as e:
            print(f"⚠️ Failed to overwrite narrative arc: {e}")
            return False

    def close(self):
        pass # Supabase handles connection pooling automatically

def generate_gemini_response(sender, receiver, context, room, db_logger):
    """Calls Gemini Pro to generate a dialogue line with retry logic."""
    profile = db_logger.get_npc_profile(sender)
    personality = profile[0] if profile else f"You are {sender}."
    
    formatted_prompt = SYSTEM_PROMPT.format(
        sender=sender,
        personality=personality,
        room=room,
        receivers=receiver,
        context=context if context else "(No previous conversation)"
    )

    # Retry logic for rate limits / temporary API glitches
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=model, contents=formatted_prompt
            )
            clean_text = response.text.strip().replace(f"{sender}:", "").replace('"', '')
            return clean_text
            
        except Exception as e:
            print(f"⚠️ API Error for {sender} (Attempt {attempt+1}/3): {e}")
            time.sleep(2 ** attempt) # Waits 1s, then 2s, then 4s

    print(f"❌ Gemini API completely failed for {sender}. Skipping turn.")
    return None

# --- CONTROLLER ---

class JerichoController:
    SAVE_DIR = "Saves"
    def __init__(self, game_file_path, session_id):
        self.env = jericho.FrotzEnv(game_file_path)
        self.session_id = session_id
        self.logger = SQLLogger(session_id)
        self.npc_names = ["Bob", "Alice", "Guard"]
        
        # 1. Ask Supabase if a state exists for this session
        saved_state_b64 = self.logger.get_zmachine_state()
        
        # 2. Cloud State Restoration Logic
        if saved_state_b64:
            self.load_game(saved_state_b64)
        else:
            self.tick_count = 0
            self.env.reset()
            print(f"🆕 Started fresh cloud simulation for session {session_id}")
            self.update_world_state()

    def save_game(self):
        """Flushes the Z-Machine state tuple to Supabase."""
        # Extract and convert the tuple into a text-safe Base64 string
        state_tuple = self.env.get_state()
        pickled_bytes = pickle.dumps(state_tuple)
        b64_state = base64.b64encode(pickled_bytes).decode('utf-8')
        
        # Push to the cloud
        self.logger.save_zmachine_state(b64_state)
        self.logger.save_current_tick(self.tick_count)
        print(f"💾 Cloud state saved for {self.session_id} at Tick {self.tick_count}")

    def load_game(self, b64_state):
        """Restores the Z-Machine state from the Supabase Base64 string."""
        try:
            # Decode the text back into binary and unpickle it
            pickled_bytes = base64.b64decode(b64_state)
            state_tuple = pickle.loads(pickled_bytes)
            
            # Inject into the emulator
            self.env.set_state(state_tuple)
            self.tick_count = self.logger.get_saved_tick()
            print(f"📂 Cloud state restored! Resuming {self.session_id} from Tick {self.tick_count}")
            
        except (EOFError, pickle.UnpicklingError, base64.binascii.Error) as e:
            print(f"⚠️ Warning: Cloud save corrupted. Starting fresh. Error: {e}")
            self.tick_count = 0
            self.env.reset()
            self.update_world_state()

    def parse_locations(self, observation_text) -> Dict[str, str]:
        locations = {}
        pattern = re.compile(r"DATA_LOC:\s*(.*?)\s*\|\s*(.*?)(?=DATA_LOC|---|$)")
        for match in pattern.finditer(observation_text):
            locations[match.group(1).strip()] = match.group(2).strip()
        return locations
    
    def parse_events(self, observation_text) -> list:
        """Safely parses DATA_EVENT strings line-by-line."""
        print("Entered parse events")
        events = []
        
        # Split the text into individual lines to isolate the logs
        for line in observation_text.splitlines():
            line = line.strip()
            
            # Only process lines that explicitly start with our tag
            if line.startswith("DATA_EVENT:"):
                # Remove the tag itself
                clean_line = line.replace("DATA_EVENT:", "", 1).strip()
                
                # Split the remaining string by the pipe delimiter
                parts = [part.strip() for part in clean_line.split("|")]
                
                # Safety check: Only add the event if we successfully extracted all 4 parts
                if len(parts) >= 4:
                    events.append({
                        "agent": parts[0],
                        "action": parts[1],
                        "target": parts[2],
                        "room": parts[3]
                    })
                else:
                    print(f"⚠️ Warning: Malformed event string from Inform: {line}")
        print(events)            
        return events

    def parse_room_data(self, observation_text):
        pattern = re.compile(
            r"DATA_ROOM:\s*(.*?)\s*\|\s*(.*?)(?=DATA_ROOM|--- END ROOM DATA ---|$)",
            re.DOTALL
        )

        count = 0
        for match in pattern.finditer(observation_text):
            r_name = match.group(1).strip()
            r_desc = match.group(2).strip()

            self.logger.update_room_desc(r_name, r_desc, self.tick_count)
            count += 1

        if count > 0:
            print(f"(Updated descriptions for {count} rooms)")

    def parse_npc_data(self, observation_text):
            pattern = re.compile(
                r"DATA_NPC:\s*(.*?)\s*\|\s*(.*?)(?=DATA_NPC|--- END NPC DATA ---|$)",
                re.DOTALL
            )
            for match in pattern.finditer(observation_text):
                npc_name = match.group(1).strip()
                raw_desc = match.group(2).strip()
                
                # If they aren't in the database yet, onboard them!
                if not self.logger.get_npc_profile(npc_name):
                    self.onboard_new_npc(npc_name, raw_desc)

    def update_world_state(self):
        """Inspects the Z-Machine memory to get the state of rooms and NPCs."""
        # 1. Update Rooms
        obs_rooms, _, _, _ = self.env.step("dump rooms")
        self.parse_room_data(obs_rooms)
        
        # 2. Update NPCs
        obs_npcs, _, _, _ = self.env.step("dump agents")
        self.parse_npc_data(obs_npcs)

    def onboard_new_npc(self, npc_name, raw_game_description):
        """Checks DB for NPC. If missing, invents their persona and caches it."""
        print(f"👤 New Character Detected: {npc_name}. Generating profile...")

        prompt = f"""
        You are a character designer for a cinematic video game.
        Character Name: "{npc_name}"
        In-game description: "{raw_game_description}"
        
        Create three things:
        1. "persona": A 2-sentence psychological profile detailing how they speak, their motivations, and attitude. Start with "You are {npc_name}..."
        2. "appearance": A strict, 1-sentence visual description for an AI image generator.
        3. "gender": Determine the gender based on the name and description. Output exactly "MALE", "FEMALE", or "NEUTRAL".
        
        Return ONLY valid JSON:
        {{
            "persona": "...",
            "appearance": "...",
            "gender": "..."
        }}
        """
        
        try:
            response = client.models.generate_content(
                model=TEXT_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            
            data = json.loads(response.text)
            persona = data.get("persona", f"You are {npc_name}.")
            appearance = data.get("appearance", f"A cinematic portrait of {npc_name}.")
            gender = data.get("gender", "NEUTRAL").upper()
            
            # ⚠️ AUTOMATED CASTING
            voice_id = cast_random_gcp_voice(gender)
            
            # Save everything to Disk!
            self.logger.save_npc_profile(npc_name, raw_game_description, persona, appearance, gender, voice_id)
            print(f"✅ Saved profile for {npc_name} (Voice: {voice_id}) to database.")
            
        except Exception as e:
            print(f"❌ Failed to generate profile for {npc_name}: {e}")
            self.logger.save_npc_profile(npc_name, raw_game_description, f"You are {npc_name}.", f"A character named {npc_name}.", "NEUTRAL", "en-US-Neural2-J")

    def conduct_group_chat(self, occupants: List[str], room: str, n_rounds=3):
        print(f"\n--- 🗣️  Group Chat in {room}: {occupants} ---")
        
        for i in range(n_rounds):
            speaker = occupants[i % len(occupants)]
            receivers = [o for o in occupants if o != speaker]
            
            ctx = self.logger.get_agent_context(speaker)
            print(f"   [Thinking...] ({speaker})")
            
            msg = generate_gemini_response(speaker, receivers, ctx, room, self.logger)
            
            # --- THE FIX: Only log and broadcast if we actually got a message ---
            if msg: 
                self.logger.log_broadcast(self.tick_count, room, speaker, receivers, msg)
                print(f"   [{speaker}] to {receivers}: {msg}")
            else:
                print(f"   [{speaker}] remains silent due to API limits.")
            
            time.sleep(1)

    def step(self):
        self.tick_count += 1
        
        obs, _, _, _ = self.env.step('step')
        self.update_world_state()
        
        # --- NEW: Parse data from the observation text ---
        locations = self.parse_locations(obs)
        events = self.parse_events(obs)
        
        # Optional: Print events to terminal so you can see the NPCs acting
        for event in events:
            print(f"🎬 [EVENT] {event['agent']} -> {event['action']} ({event['target']}) in {event['room']}")

        print(f"📍 Positions: {locations}")
        
        room_occupancy = {}
        for npc, room in locations.items():
            if room not in room_occupancy: room_occupancy[room] = []
            room_occupancy[room].append(npc)

        for room, occupants in room_occupancy.items():
            if len(occupants) >= 2:
                self.conduct_group_chat(occupants, room, n_rounds=3)
        
        # Auto-commit state after every tick
        self.save_game() 
        
        return locations

    def run(self):
        print("Starting Simulation...")
        try:
            while True:
                cmd = input(f"\n[Tick {self.tick_count}] Enter to tick, 'q' to quit > ")
                if cmd.lower() == 'q': break
                locs = self.step()
                print(f"📍 Positions: {locs}")
        finally:
            self.logger.close()

class CloudStorageManager:
    """Handles uploading, downloading, and caching files with Supabase Storage."""
    def __init__(self):
        # Automatically grab credentials and create the client
        url: str = os.getenv("SUPABASE_URL")
        key: str = os.getenv("SUPABASE_KEY")
        if not url or not key:
            raise ValueError("Supabase credentials missing in .env")
        
        self.supabase: Client = create_client(url, key)
        self.assets_bucket = 'jericho-assets'
        self.videos_bucket = 'jericho-videos'

    def file_exists(self, bucket, cloud_path):
        """Checks if a file exists in the cloud by trying to get its public URL and checking the response."""
        try:
            # list() is safer but slightly slower. A quick way is to check the folder contents.
            folder = os.path.dirname(cloud_path)
            file_name = os.path.basename(cloud_path)
            files = self.supabase.storage.from_(bucket).list(folder)
            
            # If the folder doesn't exist, it returns an empty list or error
            if files:
                for f in files:
                    if f['name'] == file_name:
                        return True
            return False
        except Exception:
            return False

    def get_public_url(self, bucket, cloud_path):
        """Returns the public URL for the frontend."""
        return self.supabase.storage.from_(bucket).get_public_url(cloud_path)

    def upload_file(self, bucket, local_path, cloud_path):
        """Uploads a local file to Supabase and returns the public URL."""
        content_type, _ = mimetypes.guess_type(local_path)
        if not content_type: content_type = 'application/octet-stream'

        with open(local_path, 'rb') as f:
            self.supabase.storage.from_(bucket).upload(
                file=f,
                path=cloud_path,
                file_options={"content-type": content_type, "upsert": "true"}
            )
        return self.get_public_url(bucket, cloud_path)

    def download_file(self, bucket, cloud_path, local_destination):
        """Downloads a cloud file to the local PC for FFmpeg or MediaPipe processing."""
        if not os.path.exists(os.path.dirname(local_destination)):
            os.makedirs(os.path.dirname(local_destination), exist_ok=True)
            
        with open(local_destination, 'wb') as f:
            res = self.supabase.storage.from_(bucket).download(cloud_path)
            f.write(res)
        return local_destination

class AutomatedDirector:
    def __init__(self):
        self.assets = {} 
        self.model_path = 'selfie_segmenter.tflite'
        self.storage = CloudStorageManager()
        self._ensure_mediapipe_model()

    def _ensure_mediapipe_model(self):
        """Downloads the MediaPipe model safely during server boot."""
        if not os.path.exists(self.model_path):
            print("⬇️ Downloading MediaPipe Selfie Segmenter model (Server Boot)...")
            url = 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite'
            import urllib.request
            urllib.request.urlretrieve(url, self.model_path)
            print("✅ MediaPipe model download complete.")
        else:
            print("✅ MediaPipe model found locally. Ready for compositing.")

    def _clean_and_parse_json(self, raw_text):
        """
        Robustly extracts JSON from an LLM response, handling 
        Markdown code blocks and extraneous text.
        """
        try:
            # 1. Try direct parsing first (fast path)
            return json.loads(raw_text)
        except json.JSONDecodeError:
            print("JSONDecodeError")
            pass

        # 2. Markdown/Text Cleaning Pattern
        # This looks for the content between the first { and the last }
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        
        if match:
            json_str = match.group(0)
            print(json_str)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"⚠️ JSON extraction failed even after cleaning: {e}")
                return None
        else:
            print("⚠️ No JSON object found in response.")
            return None

    def get_room_cache_dir(self, game_name, room_name):
        """Creates and returns a safe directory path for caching room assets."""
        safe_game = os.path.splitext(game_name)[0] # e.g., 'Control4.z8' -> 'Control4'
        safe_game = re.sub(r'\W+', '_', safe_game)
        safe_room = re.sub(r'\W+', '_', room_name)
        
        dir_path = os.path.join("Assets", safe_game, safe_room)
        os.makedirs(dir_path, exist_ok=True)
        return dir_path

    def _design_set_automatically(self, raw_room_desc, game_name, room_name):
        """Checks cache for existing set design JSON before generating."""
        cache_dir = self.get_room_cache_dir(game_name, room_name)
        json_path = os.path.join(cache_dir, "set_design.json")
        
        # 1. Check File System Cache
        if os.path.exists(json_path):
            print(f"♻️ Loading cached set design for {room_name}...")
            try:
                with open(json_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print("⚠️ Cache corrupted. Regenerating set design...")

        print(f"🏗️  AI Set Designer analyzing: '{raw_room_desc}'...")
        
        prompt = f"""
        You are a 3D environment artist. 
        I have a location description: "{raw_room_desc}".
        
        Your job is to "flesh out" this room into a strict layout.
        1. Extract the main material (walls/floor).
        2. Assign features to the North, South, East, and West walls. 
        (If the description doesn't say what's on the North wall, INVENT something that fits the theme).
        3. Determine if it is indoors or outdoors.

        Output ONLY valid JSON in this format:
        {{
            "materials": "description of walls/floor/lighting",
            "north": "description of north wall feature",
            "south": "description of south wall feature",
            "east": "description of east wall feature",
            "west": "description of west wall feature",
            "is_indoors": true/false
        }}
        """
        
        response = client.models.generate_content(
            model=TEXT_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        
        data = self._clean_and_parse_json(response.text)
        if not data or (isinstance(data, list) and len(data) == 0):
            data = {
                "materials": "Generic walls",
                "north": "Empty", "south": "Empty",
                "east": "Empty", "west": "Empty",
                "is_indoors": True
            }
        elif isinstance(data, list):
            data = data[0]

        # 2. Save to Cache
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)
            
        return data

    def extract_last_frame(self, video_path, output_image):
        """Extracts the last frame of a video using ffmpeg."""
        try:
            subprocess.run([
                "ffmpeg",
                "-y",
                "-sseof", "-0.1",
                "-i", video_path,
                "-vframes", "1",
                output_image
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            return output_image

        except Exception as e:
            print(f"❌ Failed extracting last frame: {e}")
            return None

    def create_agent_plate(self, agent_name, agent_desc, custom_prompt=None, force_recreate=False):
        """Generates an agent, uploads to cloud, and caches locally only when needed."""
        cloud_path = f"Global_Agents/{agent_name}.png"
        local_temp_path = os.path.join("Assets", "Global_Agents", f"{agent_name}.png")
        
        # 1. Cloud Cache Check
        if not force_recreate and self.storage.file_exists(self.storage.assets_bucket, cloud_path):
            print(f"♻️ Cloud Cache Hit: {agent_name}")
            return self.storage.get_public_url(self.storage.assets_bucket, cloud_path)

        # 2. LOCAL RECOVERY: Upload existing local files to bypass the API
        if not force_recreate and os.path.exists(local_temp_path):
            print(f"💾 Found existing local file for {agent_name}. Migrating to cloud...")
            try:
                return self.storage.upload_file(self.storage.assets_bucket, local_temp_path, cloud_path)
            except Exception as e:
                print(f"⚠️ Failed to migrate local asset: {e}")

        # 3. Generate via API (Will throw your 400 error if hit, but should be avoided now)
        base_desc = custom_prompt if custom_prompt else agent_desc
        prompt = f"Cinematic mid-shot of {base_desc}. The character is facing the camera. ENVIRONMENT: ISOLATED CHARACTER ON A SOLID, FLAT, LIGHT GREY BACKGROUND. Absolutely no scenery, no props, no background objects. Clean studio lighting."
        
        print(f"📸 Photographing Agent: {agent_name}...")
        try:
            os.makedirs(os.path.dirname(local_temp_path), exist_ok=True)
            response = client.models.generate_images(
                model=IMG_MODEL,
                prompt=prompt,
                config=types.GenerateImagesConfig(number_of_images=1, aspect_ratio="1:1")
            )
            response.generated_images[0].image.save(local_temp_path)
            
            # Upload to Cloud
            public_url = self.storage.upload_file(self.storage.assets_bucket, local_temp_path, cloud_path)
            return public_url
            
        except Exception as e:
            print(f"❌ Failed to photograph {agent_name}: {e}")
            return None

    def create_background_plate(self, room_data, game_name, room_name, facing_direction):
        """Generates the empty background wall and migrates local files if API is locked."""
        cloud_path = f"{game_name}/{room_name}/wall_{facing_direction}.png"
        
        # Determine where the old local script would have saved this
        cache_dir = self.get_room_cache_dir(game_name, room_name)
        local_temp_path = os.path.join(cache_dir, f"wall_{facing_direction}.png")

        # 1. Cloud Cache Check
        if self.storage.file_exists(self.storage.assets_bucket, cloud_path):
            return self.storage.get_public_url(self.storage.assets_bucket, cloud_path)

        # 2. LOCAL RECOVERY: Upload existing local backgrounds to bypass the API
        if os.path.exists(local_temp_path):
            print(f"💾 Found existing local background for {room_name} ({facing_direction}). Migrating to cloud...")
            try:
                return self.storage.upload_file(self.storage.assets_bucket, local_temp_path, cloud_path)
            except Exception as e:
                print(f"⚠️ Failed to migrate local background: {e}")

        # 3. Generate via API
        bg_feature = room_data.get(facing_direction, room_data.get('north', 'Empty'))
        prompt = f"Empty background plate, no people, no characters. Location style: {room_data['materials']}. Visible background: {bg_feature}."
        if room_data.get('is_indoors', True):
            prompt += " Environment: INTERIOR. Enclosed space. Ceiling visible. No sky."

        print(f"🖼️ Painting Background: {room_name} ({facing_direction} wall)...")
        try:
            response = client.models.generate_images(
                model=IMG_MODEL,
                prompt=prompt,
                config=types.GenerateImagesConfig(number_of_images=1, aspect_ratio="16:9")
            )
            response.generated_images[0].image.save(local_temp_path)
            
            public_url = self.storage.upload_file(self.storage.assets_bucket, local_temp_path, cloud_path)
            return public_url
        except Exception as e:
            print(f"❌ Failed to paint background: {e}")
            return None

    # --- UPDATED MEDIAPIPE COMPOSITING METHOD ---
    def composite_scene_master(self, agent_url, bg_url, game_name, room_name, agent_name, facing):
        """
        Cuts the agent out and pastes them onto the background plate using MediaPipe segmentation.
        Operates entirely as a Cloud Worker: Fetches URLs, processes locally, uploads result, and cleans up.
        """
        # 1. Define the final cloud destination path
        cloud_output_path = f"{game_name}/{room_name}/scene_master_{agent_name}_facing_{facing}.png"
        
        # 2. Check Cloud Cache First
        if self.storage.file_exists(self.storage.assets_bucket, cloud_output_path):
            print(f"♻️ Cloud Cache Hit for Composite: {agent_name} in {room_name}")
            return self.storage.get_public_url(self.storage.assets_bucket, cloud_output_path)

        print(f"✂️ Fetching assets to local PC for MediaPipe Compositing...")
        
        # 3. Define temporary local file paths for the worker node
        safe_room_name = room_name.replace(" ", "_").replace("'", "")
        temp_agent = f"temp_agent_{agent_name}.png"
        temp_bg = f"temp_bg_{safe_room_name}.png"
        temp_output = f"temp_comp_{agent_name}.png"
        
        try:
            # 4. Fetch required assets from the cloud to the local PC
            import urllib.request
            urllib.request.urlretrieve(agent_url, temp_agent)
            urllib.request.urlretrieve(bg_url, temp_bg)

            # 5. Load the downloaded images via OpenCV
            person_img = cv2.imread(temp_agent)
            bg_img = cv2.imread(temp_bg)

            if person_img is None or bg_img is None:
                print("❌ Error: Could not read downloaded images for compositing.")
                return None

            # 6. Setup MediaPipe Segmenter
            base_options = python.BaseOptions(model_asset_path=self.model_path)
            options = vision.ImageSegmenterOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                output_category_mask=False,
                output_confidence_masks=True 
            )

            with vision.ImageSegmenter.create_from_options(options) as segmenter:
                # Resize person to maintain the 85% cinematic height ratio
                bh, bw, _ = bg_img.shape
                ph, pw, _ = person_img.shape
                
                target_height = int(bh * 0.85) 
                aspect_ratio = pw / ph
                target_width = int(target_height * aspect_ratio)
                
                person_resized = cv2.resize(person_img, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)

                # Convert BGR (OpenCV) to RGB for MediaPipe
                rgb_person_img = cv2.cvtColor(person_resized, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_person_img)

                # Retrieve the segmentation mask
                segmentation_result = segmenter.segment(mp_image)
                person_mask = segmentation_result.confidence_masks[0].numpy_view()

                # Smooth the mask for a natural blend
                person_mask_blurred = cv2.GaussianBlur(person_mask, (7, 7), 0)
                mask_3d = np.stack((person_mask_blurred,) * 3, axis=-1)

                # Composite the images using Alpha Blending
                x_offset = (bw - target_width) // 2
                y_offset = bh - target_height

                roi = bg_img[y_offset:y_offset+target_height, x_offset:x_offset+target_width]

                foreground = (person_resized * mask_3d).astype(np.uint8)
                background = (roi * (1 - mask_3d)).astype(np.uint8)
                blended = cv2.add(foreground, background)

                # Overwrite the Region of Interest in the background image with the blended result
                bg_img[y_offset:y_offset+target_height, x_offset:x_offset+target_width] = blended

                # Save the final stitched image to our temporary output file
                cv2.imwrite(temp_output, bg_img)

            # 7. Upload the stitched result back to the cloud
            public_url = self.storage.upload_file(self.storage.assets_bucket, temp_output, cloud_output_path)
            print(f"✅ Success! Uploaded composite to {public_url}")
            return public_url
            
        except Exception as e:
            print(f"❌ Compositing failed: {e}")
            return None
            
        finally:
            # 8. Clean up local worker space
            # This 'finally' block executes no matter what happens above, keeping your disk clean.
            for temp_file in [temp_agent, temp_bg, temp_output]:
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except OSError:
                        pass

    def prepare_scene_assets(self, room_description, agents, game_name, room_name):
        """Phase 1: Generates the master composited images using your MediaPipe pipeline."""
        room_data = self._design_set_automatically(room_description, game_name, room_name)
        
        scene_assets = {}
        for agent in agents:
            name = agent['name']
            desc = agent['desc']
            facing = agent.get('facing', 'north').lower()
            
            # Generate the individual plates
            agent_path = self.create_agent_plate(name, desc)
            bg_path = self.create_background_plate(room_data, game_name, room_name, facing)
            
            # Composite them using your MediaPipe Segmenter!
            if agent_path and bg_path:
                comp_path = self.composite_scene_master(agent_path, bg_path, game_name, room_name, name, facing)
                if comp_path:
                    scene_assets[name] = comp_path
                
        return scene_assets
    
    def _save_transcript(self, script, video_filename):
        """Saves a .txt transcript alongside the generated video."""
        # Replace .mp4 with .txt safely, ensuring the exact same ID
        script_path = video_filename.rsplit('.', 1)[0] + '.txt'
        try:
            with open(script_path, "w", encoding="utf-8") as f:
                f.write("--- SCENE TRANSCRIPT ---\n\n")
                for speaker, line, emotion, _ in script:
                    clean_line = line.strip()
                    f.write(f"[{emotion.upper()}] {speaker}: {clean_line}\n\n")
            print(f"📄 Transcript saved to {script_path}")
        except Exception as e:
            print(f"⚠️ Failed to save transcript for {video_filename}: {e}")

    def film_scene(self, script, scene_assets, game_name, room_name, output_filename, continuity=False):
        """Phase 2: Sends the cached MediaPipe composites to Veo and syncs TTS."""
        
        self._save_transcript(script, output_filename)
        clips = []
        last_frame_path = None

        for i, (speaker, line, emotion, voice_id) in enumerate(script):
            
            line = line.strip()
            is_action = line.startswith("*") and line.endswith("*")

            # 1. Prepare the Veo Prompt
            if is_action:
                # It's an action! Remove asterisks and tell Veo not to move lips.
                action = line.strip("*")
                text_prompt = f"""
                Cinematic mid-shot.
                The character looks {emotion}.
                The character performs the following action: {action}.
                The character DOES NOT SPEAK. Their mouth remains closed.
                Cinematic lighting, realistic animation.
                """
            else:
                # It's normal dialogue.
                camera_angle = "Cinematic mid-shot" if i % 2 == 0 else "Cinematic close-up"
                text_prompt = f"""
                {camera_angle}.
                The character looks {emotion}.
                The character speaks the following line clearly: "{line}"
                Cinematic lighting, realistic facial animation.
                """

            # Choose input image
            input_image_path = scene_assets[speaker]

            if continuity and last_frame_path:
                input_image_path = last_frame_path

            with open(input_image_path, "rb") as f:
                raw_data = f.read()

            image_input = types.Image(image_bytes=raw_data, mime_type="image/png")

            try:
                # 2. Generate Video with Veo
                operation = client.models.generate_videos(
                    model=VID_MODEL,
                    prompt=text_prompt,
                    image=image_input,
                    config=types.GenerateVideosConfig(number_of_videos=1, aspect_ratio="16:9")
                )

                attempts = 0
                while not operation.done and attempts < 60:
                    time.sleep(5)
                    operation = client.operations.get(operation)
                    attempts += 1

                if operation.result and operation.result.generated_videos:
                    raw_vid_fname = f"raw_clip_{i:03d}_{speaker}.mp4"
                    downloaded = self._download_video(
                        operation.result.generated_videos[0].video.uri,
                        raw_vid_fname
                    )

                    if downloaded:
                        tts_fname = f"audio_{i:03d}_{speaker}.mp3"
                        final_sync_fname = f"sync_clip_{i:03d}_{speaker}.mp4"
                        
                        # 3. Generate the TTS Audio
                        audio_success = self.generate_tts_audio(speaker, line, tts_fname, voice_id)
                        
                        # 4. Sync Audio and Video (or fallback for silent actions)
                        if audio_success and os.path.exists(tts_fname):
                            self._sync_audio_video(downloaded, tts_fname, final_sync_fname)
                            clips.append(final_sync_fname)
                            
                            # Clean up intermediate files
                            if os.path.exists(downloaded): os.remove(downloaded)
                            if os.path.exists(tts_fname): os.remove(tts_fname)
                        else:
                            # If it's an action line or API failed, just use raw video
                            clips.append(downloaded)
                            final_sync_fname = downloaded 

                        # 5. Extract last frame for continuity
                        if continuity:
                            frame_path = final_sync_fname.replace(".mp4", "_lastframe.png")
                            last_frame = self.extract_last_frame(final_sync_fname, frame_path)
                            if last_frame:
                                last_frame_path = last_frame

            except Exception as e:
                print(f"❌ Clip failed: {e}")

        # 6. Stitch the final scene
        if clips:
            print(f"🎞️ Stitching {len(clips)} clips...")

            with open(f"ffmpeg_list_{room_name}.txt", "w") as f:
                for vid in clips:
                    f.write(f"file '{vid}'\n")

            subprocess.run([
                "ffmpeg",
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", f"ffmpeg_list_{room_name}.txt",
                "-c", "copy",
                output_filename
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            os.remove(f"ffmpeg_list_{room_name}.txt")

            for vid in clips:
                if os.path.exists(vid):
                    os.remove(vid)

            print(f"🍿 Scene Complete: {output_filename}")
            return output_filename

        return None
    
    def film_scene_stills(self, script, scene_assets, game_name, room_name, output_filename):
        """Phase 2 Alternative: Generates an animatic using static images and TTS (No Veo)."""
        
        self._save_transcript(script, output_filename)
        
        clips = []

        for i, (speaker, line, emotion, voice_id) in enumerate(script):
            line = line.strip()
            is_action = line.startswith("*") and line.endswith("*")
            
            input_image_path = scene_assets[speaker]
            final_sync_fname = f"animatic_clip_{i:03d}_{speaker}.mp4"

            try:
                if is_action:
                    # Action lines: Just show the character silently for 3 seconds
                    subprocess.run([
                        "ffmpeg", "-y", "-loop", "1", "-i", input_image_path,
                        "-c:v", "libx264", "-t", "3", "-pix_fmt", "yuv420p",
                        final_sync_fname
                    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    clips.append(final_sync_fname)
                else:
                    tts_fname = f"audio_{i:03d}_{speaker}.mp3"
                    
                    # Generate the TTS Audio
                    audio_success = self.generate_tts_audio(speaker, line, tts_fname, voice_id)
                    
                    if audio_success and os.path.exists(tts_fname):
                        # Combine looped image with audio
                        subprocess.run([
                            "ffmpeg", "-y", "-loop", "1", "-i", input_image_path,
                            "-i", tts_fname,
                            "-c:v", "libx264", "-c:a", "aac", "-b:a", "192k",
                            "-pix_fmt", "yuv420p", "-shortest",
                            final_sync_fname
                        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        
                        clips.append(final_sync_fname)
                        if os.path.exists(tts_fname): os.remove(tts_fname)
                    else:
                        # Fallback: 3 seconds of silence if TTS fails
                        subprocess.run([
                            "ffmpeg", "-y", "-loop", "1", "-i", input_image_path,
                            "-c:v", "libx264", "-t", "3", "-pix_fmt", "yuv420p",
                            final_sync_fname
                        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        clips.append(final_sync_fname)

            except Exception as e:
                print(f"❌ Animatic Clip failed: {e}")

        # Stitch the final animatic scene
        if clips:
            print(f"🎞️ Stitching {len(clips)} animatic clips...")
            list_file = f"ffmpeg_list_{room_name}_animatic.txt"
            
            with open(list_file, "w") as f:
                for vid in clips:
                    f.write(f"file '{vid}'\n")

            subprocess.run([
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", list_file, "-c", "copy", output_filename
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            os.remove(list_file)
            for vid in clips:
                if os.path.exists(vid): os.remove(vid)

            print(f"🍿 Animatic Complete: {output_filename}")
            return output_filename

        return None
        
    def _download_video(self, video_uri, filename):
        # Helper to download video from URI
        headers = {"x-goog-api-key": GEMINI_API_KEY}
        response = requests.get(video_uri, headers=headers, stream=True)
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"✅ Cut! Saved to {filename}")
            return filename
        return None

    def generate_recap_script(self, timeline_data):
        """Converts raw scene data into a dramatic Narrator script."""
        
        # 1. Format the history for the LLM
        raw_log = ""
        for scene in timeline_data:
            raw_log += f"--- Room: {scene['room']} (Tick {scene['tick']}) ---\n"
            for line in scene['script']:
                raw_log += f"{line['speaker']}: {line['line']}\n"

        if not raw_log:
            return [("Narrator", "The story has just begun...", "neutral")]

        # 2. Ask Gemini to write a recap
        prompt = f"""
        You are the dramatic narrator of a cinematic web series.
        Read the following transcript of the previous scenes:
        
        {raw_log}
        
        Write a short, engaging "Previously on..." recap monologue.
        Keep it under 3 sentences. Focus on the core conflict or mystery.
        Do not add sound effects or stage directions. Output ONLY the spoken text.
        """
        
        print("✍️ Writing the recap script...")
        response = client.models.generate_content(
            model=TEXT_MODEL,
            contents=prompt
        )
        
        recap_text = response.text.strip()
        
        # Return it in the exact format your film_scene method expects
        return [("Narrator", f"Previously on Jericho... {recap_text}", "serious")]

    def get_fixed_narrator_composite(self):
        """Creates or loads the permanent Narrator asset."""
        # Use your existing directory generator so the paths match perfectly
        studio_dir = self.get_room_cache_dir("Global_Agents", "Narrator_Studio")
        
        # This is the EXACT filename composite_scene_master will generate
        expected_output_name = "scene_master_Narrator_facing_north.png"
        comp_path = os.path.join(studio_dir, expected_output_name)
        
        # If it's already generated, skip everything
        if os.path.exists(comp_path):
            print("♻️ Loading existing Narrator studio...")
            return comp_path
            
        print("🎙️ Building the permanent Narrator studio...")
        
        # 1. Generate the Agent
        agent_path = os.path.join(studio_dir, "Narrator_Raw.png")
        if not os.path.exists(agent_path):
            prompt = "Cinematic mid-shot of a mysterious, sharply dressed narrator standing confidently. Solid grey background."
            res = client.models.generate_images(model=IMG_MODEL, prompt=prompt, config=types.GenerateImagesConfig(number_of_images=1, aspect_ratio="1:1"))
            res.generated_images[0].image.save(agent_path)

        # 2. Generate the Background
        bg_path = os.path.join(studio_dir, "Narrator_BG.png")
        if not os.path.exists(bg_path):
            prompt = "Empty background. A dark, cinematic broadcast studio with subtle neon backlighting. No people."
            res = client.models.generate_images(model=IMG_MODEL, prompt=prompt, config=types.GenerateImagesConfig(number_of_images=1, aspect_ratio="16:9"))
            res.generated_images[0].image.save(bg_path)

        # 3. Composite! We return the direct output of this function so we know the path is 100% accurate.
        return self.composite_scene_master(
            agent_path=agent_path, 
            bg_path=bg_path, 
            game_name="Global_Agents", 
            room_name="Narrator_Studio", 
            agent_name="Narrator",  # Matches the expected_output_name above
            facing="north"
        )
    
    def generate_recap_video(self, db_logger, game_name, past_n=3, output_filename="recap.mp4"):
        """Fetches history, summarizes it, and films the Narrator."""
        
        # 1. Fetch History
        timeline_data = db_logger.get_timeline_history(past_n)
        if not timeline_data:
            print("⚠️ No events in the timeline to recap.")
            return None

        # 2. Format history for the LLM
        raw_log = ""
        for scene in timeline_data:
            raw_log += f"--- Room: {scene['room']} (Tick {scene['tick']}) ---\n"
            for line in scene['script']:
                raw_log += f"{line['speaker']}: {line['line']}\n"

        # 3. Write the Script
        prompt = f"""
        You are the dramatic narrator of a cinematic web series.
        Read the following transcript of the previous scenes:
        {raw_log}
        Write a short, engaging "Previously on..." recap monologue.
        Keep it under 3 sentences. Output ONLY the spoken text. No stage directions.
        """
        response = client.models.generate_content(model=TEXT_MODEL, contents=prompt)
        recap_text = f"Previously, in {game_name}... " + response.text.strip()
        
        # Format the script exactly how film_scene expects it: [(speaker, line, emotion)]
        script = []
        sentences = split_into_sentences(recap_text)
        for sentence in sentences:
            script.append(("Narrator", sentence, "serious"))

        # 4. Grab Fixed Asset
        narrator_img = self.get_fixed_narrator_composite()
        scene_assets = {"Narrator": narrator_img}

        # 5. Film & Stitch (Reusing your robust pipeline!)
        # 5. Film & Stitch (Using continuity mode)
        print("🎬 Filming Recap Video...")

        return self.film_scene(
            script,
            scene_assets,
            game_name,
            "Recap_Studio",
            output_filename,
            continuity=True
        )

    def generate_tts_audio(self, speaker, text, output_filename, voice_name):
        """Generates TTS using Google Cloud TTS and saves it as an .mp3."""
        if text.strip().startswith("*") and text.strip().endswith("*"):
            print(f"🔇 Skipping TTS for action: {text}")
            return False

        api_key = os.environ.get("GCP_API_KEY", "")
        if not api_key:
            print("❌ Error: GCP_API_KEY not found in environment.")
            return False

        if not voice_name:
            voice_name = "en-US-Neural2-J" 

        language_code = "-".join(voice_name.split("-")[:2])

        url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={api_key}"
        payload = {
            "input": {"text": text},
            "voice": {"languageCode": language_code, "name": voice_name},
            "audioConfig": {"audioEncoding": "MP3"}
        }

        print(f"🎙️ Recording GCP TTS for {speaker} ({voice_name})...")
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status() 
            
            response_data = response.json()
            audio_content_b64 = response_data.get("audioContent")
            
            if audio_content_b64:
                audio_bytes = base64.b64decode(audio_content_b64)
                with open(output_filename, 'wb') as f:
                    f.write(audio_bytes)
                print(f"✅ Audio saved to {output_filename}")
                return True
            else:
                return False
                
        except Exception as e:
            print(f"❌ GCP API Error: {e}")
            return False

    def _get_media_duration(self, file_path):
        """Returns the duration of a media file in seconds using ffprobe."""
        try:
            result = subprocess.run([
                "ffprobe", "-v", "error", "-show_entries",
                "format=duration", "-of",
                "default=noprint_wrappers=1:nokey=1", file_path
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
            return float(result.stdout.strip())
        except Exception as e:
            print(f"❌ Failed to get duration for {file_path}: {e}")
            return 0.0

    def _sync_audio_video(self, video_path, audio_path, output_path):
        """Applies heuristics to sync TTS audio with generated video."""
        vid_dur = self._get_media_duration(video_path)
        aud_dur = self._get_media_duration(audio_path)

        if vid_dur == 0 or aud_dur == 0:
            print("⚠️ Invalid media durations, falling back to basic overlay.")
            subprocess.run(["ffmpeg", "-y", "-i", video_path, "-i", audio_path, "-c:v", "copy", "-c:a", "aac", "-shortest", output_path], check=True)
            return

        # The 15% Heuristic
        difference_ratio = abs(vid_dur - aud_dur) / aud_dur

        print(f"⚖️ Syncing: Video is {vid_dur:.2f}s, Audio is {aud_dur:.2f}s (Diff: {difference_ratio:.1%})")

        try:
            if difference_ratio <= 0.15:
                # Scenario A: Stretch/Compress Audio (Less than 15% difference)
                print("   ➔ Difference is small. Adjusting audio tempo.")
                speed_factor = aud_dur / vid_dur 
                subprocess.run([
                    "ffmpeg", "-y", 
                    "-i", video_path, "-i", audio_path, 
                    "-c:v", "copy", 
                    "-af", f"atempo={speed_factor}", 
                    "-map", "0:v:0", "-map", "1:a:0", 
                    output_path
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            elif vid_dur > aud_dur:
                # Scenario B: Video is much longer than audio -> Cut video early
                print("   ➔ Video is too long. Cutting video to match audio length.")
                subprocess.run([
                    "ffmpeg", "-y", 
                    "-i", video_path, "-i", audio_path, 
                    "-c:v", "copy", "-c:a", "aac", 
                    "-map", "0:v:0", "-map", "1:a:0", 
                    "-shortest", 
                    output_path
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            else:
                # Scenario C: Audio is much longer than video -> Freeze last frame of video
                print("   ➔ Audio is too long. Freezing last frame of video.")
                subprocess.run([
                    "ffmpeg", "-y", 
                    "-i", video_path, "-i", audio_path, 
                    "-filter_complex", "[0:v]tpad=stop_mode=clone:stop_duration=10[v]", 
                    "-map", "[v]", "-map", "1:a:0", 
                    "-c:v", "libx264", "-c:a", "aac", 
                    "-shortest", 
                    output_path
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        except subprocess.CalledProcessError as e:
            print(f"❌ FFmpeg sync failed: {e}")

    def spice_up_story(self, timeline_data, db_logger):
        """
        Executes the untethered 3-Step Writers' Room Pipeline.
        The AI determines the final scene count and pacing.
        """
        if not timeline_data:
            return {"error": "The timeline is empty."}

        known_rooms = set()
        known_chars = set()
        raw_log = []
        
        # We need the boundaries to know what chunk of the database to replace
        ticks = [scene['tick'] for scene in timeline_data]
        min_tick, max_tick = min(ticks), max(ticks)
        
        # 1. Gather all raw context
        for scene in timeline_data:
            known_rooms.add(scene['room'])
            scene_obj = {
                "original_tick": scene['tick'],
                "room": scene['room'],
                "script": [{"speaker": line['speaker'], "line": line['line']} for line in scene['script']]
            }
            for line in scene['script']:
                known_chars.add(line['speaker'])
            raw_log.append(scene_obj)

        allowed_rooms_str = ", ".join(known_rooms)
        allowed_chars_str = ", ".join(known_chars)

        char_context = ""
        for char in known_chars:
            profile = db_logger.get_npc_profile(char)
            if profile:
                char_context += f"- {char}: {profile[0]}\n"

        # 2. The Untethered Showrunner Prompt
        prompt = f"""
        You are the Lead Showrunner for a cinematic series. You must process the raw simulation logs into a tightly paced, highly dramatic episode.
        
        ALLOWED CAST: [{allowed_chars_str}]
        CAST PERSONAS:
        {char_context}
        ALLOWED LOCATIONS: [{allowed_rooms_str}]
        
        RAW SIMULATION LOGS (For inspiration only):
        {json.dumps(raw_log)}
        
        You must output a strictly formatted JSON object that follows a 3-step creative pipeline:
        
        STEP 1: "plot_setup" 
        Define the current stakes based on the raw logs. What is the overarching mystery, goal, or threat connecting these characters right now?
        
        STEP 2: "birds_eye_view"
        Write a compressed plot outline. How does the story progress? What is the narrative arc?
        
        STEP 3: "scenes"
        Write the actual dialogue. 
        CRITICAL INSTRUCTION: DO NOT copy the pacing of the raw logs. You have total freedom to restructure the timeline. If the raw logs took 10 boring steps to do something, condense it into 2 or 3 highly dramatic scenes. 
        - You decide how many scenes are needed to tell the story from Step 2.
        - You decide which allowed characters are in which allowed rooms for each scene.
        - Dialogue MUST be purposeful. No aimless bickering.
        
        Output ONLY valid JSON matching this exact structure:
        {{
            "plot_setup": "...",
            "birds_eye_view": "...",
            "scenes": [
                {{
                    "room": "Kitchen",
                    "script": [
                        {{"speaker": "Alice", "line": "..."}},
                        {{"speaker": "Bob", "line": "*looks around nervously* ..."}}
                    ]
                }}
            ]
        }}
        """
        
        print("✍️ The Writers' Room is breaking the story and restructuring the timeline...")
        try:
            response = client.models.generate_content(
                model=TEXT_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            
            master_script = json.loads(response.text)
            
            print("\n=== AI SHOWRUNNER NOTES ===")
            print(f"🎬 PLOT SETUP:\n{master_script.get('plot_setup', 'None')}\n")
            print(f"🗺️ OUTLINE:\n{master_script.get('birds_eye_view', 'None')}\n")
            print("===========================\n")
            
            new_storyline = master_script.get("scenes", [])
            
            # Execute the Arc Replacement!
            success = db_logger.overwrite_narrative_arc(
                min_tick=min_tick, 
                max_tick=max_tick, 
                new_scenes=new_storyline
            )
            
            if success:
                return {"success": f"Timeline restructured! Compressed {len(timeline_data)} simulation ticks into {len(new_storyline)} cinematic scenes."}
            else:
                return {"error": "Failed to save the new arc to the database."}
            
        except Exception as e:
            print(f"❌ Failed to write the script: {e}")
            return {"error": f"Failed to generate the script: {str(e)}"}

class SceneSelector:

    def __init__(self, db, director, key_terms=None,game_name="UnknownGame"):
        self.db = db
        self.director = director
        self.room_asset_cache = {}
        self.game_name = game_name 
    
        # INTEGRATION: Initialize the Drama Engine
        # We pass your specific game terms here
        self.scorer = DramaScorer(key_terms=key_terms) 

    def scan_and_rank(self, start_tick, end_tick):
        """
        Scans a range of ticks and returns the single most dramatic moment 
        for EACH tick, creating a sequence of 'Dailies'.
        """
        print(f"🕵️  Scanning ticks {start_tick} to {end_tick} for daily dailies...")
        
        dailies = []

        for tick in range(start_tick, end_tick + 1):
            rooms = self.db.get_rooms_for_tick(tick)
            if not rooms:
                continue

            best_score = -1
            best_scene_data = None
            best_room = None

            for room in rooms:
                scene_data = self.db.get_structured_scene_data(tick, room)
                script_data = scene_data.get("script", [])

                if not script_data or len(script_data) < 2:
                    continue  

                scorer_input = [
                    {"speaker": row["speaker"], "text": row["line"]} 
                    for row in script_data
                ]

                score = self.scorer.calculate_score(scorer_input)

                # Find the most dramatic room for this specific tick
                if score > best_score:
                    best_score = score
                    best_scene_data = scene_data
                    best_room = room

            # Only add to dailies if something moderately interesting happened
            if best_score > 10:
                dailies.append((best_score, tick, best_room, best_scene_data))

        print(f"\n🎬 FOUND {len(dailies)} DAILIES FOR REVIEW")
        return dailies

    def run_auto(self, start_tick, end_tick, selection="all"):
        """
        Automated workflow updated for server use (no input() prompts).
        selection can be "all" or an integer index.
        """
        top_scenes = self.scan_and_rank(start_tick, end_tick)
        
        if not top_scenes:
            print("No interesting scenes found in this range.")
            return

        # Server-safe selection logic
        if str(selection).lower() == 'all':
            for _, _, _, scene_data in top_scenes:
                self._generate_video_from_scene(scene_data)
        else:
            try:
                idx = int(selection) - 1
                if 0 <= idx < len(top_scenes):
                    _, _, _, scene_data = top_scenes[idx]
                    self._generate_video_from_scene(scene_data)
                else:
                    print(f"⚠️ Selection {selection} out of range.")
            except ValueError:
                print(f"⚠️ Invalid selection parameter: {selection}")

    def _detect_emotion_nrc(self,sentence, threshold=0.3):
        emotion = NRCLex(sentence)
        raw_scores = emotion.raw_emotion_scores

        # If no emotions detected
        if not raw_scores:
            print("No raw scores!!!")
            return "neutral"
        
        # Normalize scores
        total = sum(raw_scores.values())
        normalized_scores = {k: v / total for k, v in raw_scores.items()}
        
        # Get highest emotion
        dominant_emotion = max(normalized_scores, key=normalized_scores.get)
        print(dominant_emotion)
        max_score = normalized_scores[dominant_emotion]
        
        if max_score >= threshold:
            label = dominant_emotion
        else:
            label = "neutral"
        
        return label

    def _generate_video_from_scene(self, scene_data):
        room_name = scene_data["room"]
        room_visual = scene_data["visual"]
        script_data = scene_data["script"]

        if not script_data:
            return

        script = []
        agents = []
        speakers = set()

        for line in script_data:
            speaker = line["speaker"]
            text = line["line"]
            
            emotion = self._detect_emotion_nrc(text)
            voice_id = self.db.get_npc_voice(speaker)
            script.append((speaker, text, emotion, voice_id))
            
            if speaker not in speakers:
                agents.append({
                    "name": speaker,
                    "desc": f"{speaker}, a character in the Jericho world",
                    "facing": "north"
                })
                speakers.add(speaker)

        print(f"🎥 Rendering Tick {scene_data.get('tick')} for {self.game_name}...")
        
        # ⚠️ We completely removed the old `if room_name not in self.room_asset_cache:` block!
        # The director handles all the set design and asset caching internally now.
        
        self.director.produce_scene(
            room_description=room_visual,
            agents=agents,
            script=script,
            game_name=self.game_name,
            room_name=room_name,
            output_filename=f"scene_{self.game_name}_tick_{scene_data.get('tick', 0)}_{room_name}.mp4"
        )

if __name__ == "__main__":
    try:
        director = AutomatedDirector()
        GAME_FILE = "Control4.z8"  
        db_manager = GameDBManager()
        try:
            print("Fetching Character voice...")
            fetch_gcp_casting_menu()
        except Exception as e:
            print("Exception in fetching voice:")
            print(e)
        db_name = db_manager.get_or_create_db(GAME_FILE)
        controller = JerichoController(game_file_path=GAME_FILE,db_name=db_name)
        video_logger = SQLLogger(db_name)
        controller.run()
        print(video_logger.get_rooms_for_tick(1))
        game_terms = ["darkness", "grue", "lamp", "spell"]
        videomaker = SceneSelector(db=video_logger,director=director,key_terms=game_terms)
        videomaker.run_auto(start_tick=0, end_tick=500)
        
    except FileNotFoundError:
        print(f"Error: Could not find game file '{GAME_FILE}'")

    except FileNotFoundError:
        print("Game file not found.")