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

# --- NEW IMPORTS FOR MEDIAPIPE ---
import cv2
import numpy as np
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)
model = "gemini-2.0-flash"
vid_model = "veo-3.1-fast-generate-preview"
IMG_MODEL = "imagen-4.0-fast-generate-001"
VID_MODEL = "veo-3.1-fast-generate-preview"
TEXT_MODEL = "gemini-2.0-flash"
video_client = genai.Client(api_key=GEMINI_API_KEY)

NPC_PERSONAS = {
    "Bob": "You are Bob, a paranoid scientist. You are suspicious of everyone.",
    "Alice": "You are Alice, a clumsy intern trying to act professional.",
    "Guard": "You are a grumpy security guard who hates noise.",
    "DEFAULT": "You are a curious inhabitant of this world."
}

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


NPC_APPEARANCE = {
    "Bob": "A middle-aged scientist wearing a white lab coat, messy hair, glasses, nervous expression",
    "Alice": "A young intern in business casual attire, holding a clipboard, looking slightly confused",
    "Guard": "A burly security guard in a dark blue uniform, flashlight on belt, stern face",
    "DEFAULT": "A mysterious figure in shadow"
}

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

    def get_or_create_db(self, game_file):
        game_file = os.path.basename(game_file)

        # 1. Check registry
        with open(self.REGISTRY_FILE, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row:
                    continue
                if row["game_file"] == game_file:
                    return row["db_name"]

        # 2. If not found → create new DB entry prepended with the folder path
        raw_db_name = f"db_{os.path.splitext(game_file)[0]}.db"
        db_name = os.path.join(self.DB_FOLDER, raw_db_name)

        with open(self.REGISTRY_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([game_file, db_name])

        print(f"🆕 Created new DB mapping: {game_file} → {db_name}")
        
        # --- THE FIX: Actually create the physical DB file and base schema ---
        import sqlite3
        try:
            conn = sqlite3.connect(db_name)
            cursor = conn.cursor()
            # Initialize the base table so the file is written to disk properly
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS room_desc (
                room_name TEXT,
                description TEXT,
                tick INTEGER,
                PRIMARY KEY (room_name, tick)
            );
            """)
            conn.commit()
            conn.close()
            print(f"🗄️ Initialized physical SQLite database at: {db_name}")
        except sqlite3.Error as e:
            print(f"❌ Failed to initialize database file {db_name}: {e}")

        return db_name
    def sync_games_directory(self, games_dir):
        """Scans the games folder on boot and ensures all games are registered."""
        print(f"🔄 Syncing registry with {games_dir}...")
        if not os.path.exists(games_dir):
            return
            
        for filename in os.listdir(games_dir):
            file_path = os.path.join(games_dir, filename)
            # Only register actual files (ignoring subdirectories or hidden OS files)
            if os.path.isfile(file_path) and not filename.startswith('.'):
                # get_or_create_db already checks if it exists, so this is safe to call
                self.get_or_create_db(filename)
        print("✅ Registry sync complete.")

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
    def __init__(self, db_name="jericho_game.db"):
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.init_room_desc_table()

    def sanitize_name(self, name):
        clean_name = re.sub(r'\W+', '_', name)
        return f"Table_{clean_name}"

    def sanitize_room_name(self,name):
        clean_name = re.sub(r'\W+', '_', name)
        return f"RoomTable_{clean_name}"

    def init_room_desc_table(self):
        sql = """
        CREATE TABLE IF NOT EXISTS room_desc (
            room_name TEXT,
            description TEXT,
            tick INTEGER,
            PRIMARY KEY (room_name, tick)
        );
        """
        self.cursor.execute(sql)
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS official_timeline (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tick INTEGER,
            room_name TEXT,
            UNIQUE(tick, room_name)
        );
        """)
        self.conn.commit()

    def update_room_desc(self, room_name, description, tick):
        # We now store a snapshot for every tick
        sql = "INSERT OR REPLACE INTO room_desc (room_name, description, tick) VALUES (?, ?, ?)"
        self.cursor.execute(sql, (room_name, description, tick))
        self.conn.commit()

    def ensure_table(self, table_name, is_room_log=False):
        if is_room_log:
            sql = f"""CREATE TABLE IF NOT EXISTS {table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tick INTEGER,
                sender TEXT,
                receiver TEXT,
                message TEXT
            );"""
        else:
            sql = f"""CREATE TABLE IF NOT EXISTS {table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tick INTEGER,
                room TEXT,
                sender TEXT,
                receiver TEXT,
                message TEXT
            );"""
        self.cursor.execute(sql)
        self.conn.commit()

    def log_broadcast(self, tick, room, sender, receivers, message):
        """
        Logs a message from one Sender to N Receivers.
        """
        try:
            # 1. LOG TO ROOM (The 'Public' Record)
            # Receiver is marked as "All" or the list of names for clarity
            room_table = self.sanitize_room_name(room)
            self.ensure_table(room_table, is_room_log=True)
            recipients_str = ", ".join(receivers)
            self.cursor.execute(
                f"INSERT INTO {room_table} (tick, sender, receiver, message) VALUES (?, ?, ?, ?)",
                (tick, sender, recipients_str, message)
            )

            # 2. LOG TO SENDER (Personal Memory)
            sender_table = self.sanitize_name(sender)
            self.ensure_table(sender_table, is_room_log=False)
            self.cursor.execute(
                f"INSERT INTO {sender_table} (tick, room, sender, receiver, message) VALUES (?, ?, ?, ?, ?)",
                (tick, room, sender, recipients_str, message)
            )

            # 3. LOG TO EACH RECEIVER (Broadcasting)
            for recipient in receivers:
                rec_table = self.sanitize_name(recipient)
                self.ensure_table(rec_table, is_room_log=False)
                # Note: Sender is the speaker, Receiver is "Me" (the recipient)
                self.cursor.execute(
                    f"INSERT INTO {rec_table} (tick, room, sender, receiver, message) VALUES (?, ?, ?, ?, ?)",
                    (tick, room, sender, recipient, message)
                )

            self.conn.commit()
            
        except sqlite3.Error as e:
            print(f"❌ Database Error: {e}")

    def get_agent_context(self, agent_name, limit=5):
        table_name = self.sanitize_name(agent_name)
        try:
            self.cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?", 
                (table_name,)
            )
            if not self.cursor.fetchone(): return ""

            query = f"SELECT sender, message, tick FROM {table_name} ORDER BY id DESC LIMIT ?"
            self.cursor.execute(query, (limit,))
            rows = self.cursor.fetchall()
            rows.reverse()
            
            return "\n".join([f"[Tick {r[2]}] {r[0]}: {r[1]}" for r in rows])
        except sqlite3.Error:
            return ""

    def get_video_generation_context(self, tick, room_name):
        """
        Fetches the visual description and dialogue for a specific room and tick.
        Returns a formatted string suitable for a Video AI prompt.
        """
        # 1. Fetch Visual Description (The "Set")
        self.cursor.execute(
            "SELECT description FROM room_desc WHERE room_name = ? AND tick = ?", 
            (room_name, tick)
        )
        row = self.cursor.fetchone()
        if row:
            visual_desc = row[0]
        else:
            # Fallback: Get the most recent description before this tick
            self.cursor.execute(
                "SELECT description FROM room_desc WHERE room_name = ? AND tick < ? ORDER BY tick DESC LIMIT 1",
                (room_name, tick)
            )
            fallback = self.cursor.fetchone()
            visual_desc = fallback[0] if fallback else "An empty room."

        # 2. Fetch Dialogue (The "Script")
        table_name = self.sanitize_room_name(room_name)
        
        # Check if table exists first
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        if not self.cursor.fetchone():
            return f"Scene Description: {visual_desc}\n\n(No dialogue recorded.)"

        self.cursor.execute(
            "SELECT sender, message FROM {} WHERE tick = ?".format(table_name), 
            (tick,)
        )
        chat_rows = self.cursor.fetchall()
        
        # 3. Format the Output
        dialogue_script = ""
        for sender, msg in chat_rows:
            dialogue_script += f"- {sender}: \"{msg}\"\n"
            
        if not dialogue_script:
            dialogue_script = "(The characters are silent, looking around.)"

        final_context = (
            f"--- SCENE SETTING (Room: {room_name}) ---\n"
            f"Visual Details: {visual_desc}\n\n"
            f"--- ACTION & DIALOGUE (Tick {tick}) ---\n"
            f"{dialogue_script}"
        )
        
        return final_context

    def get_structured_scene_data(self, tick, room_name):
        """
        Returns a dictionary with separated visual context and dialogue lines.
        Used for programmatic video generation.
        """

        # 1. Fetch Room Visuals
        self.cursor.execute(
            "SELECT description FROM room_desc WHERE room_name = ? AND tick = ?", 
            (room_name, tick)
        )
        row = self.cursor.fetchone()
        
        # Fallback if specific tick is missing (use most recent)
        if not row:
            self.cursor.execute(
                "SELECT description FROM room_desc WHERE room_name = ? AND tick < ? ORDER BY tick DESC LIMIT 1",
                (room_name, tick)
            )
            row = self.cursor.fetchone()
        print(row)
        visual_desc = row[0] if row else "A dark, empty room."

        # 2. Fetch Dialogue
        table_name = self.sanitize_room_name(room_name)
        
        # Check if table exists
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        if not self.cursor.fetchone():
            return {"visual": visual_desc, "script": []}

        self.cursor.execute(
            "SELECT sender, message FROM {} WHERE tick = ? ORDER BY id ASC".format(table_name), 
            (tick,)
        )
        
        # Format as list of dicts
        script = [{"speaker": row[0], "line": row[1]} for row in self.cursor.fetchall()]
        print(script)
        print(room_name)
        return {
            "visual": visual_desc,
            "script": script,
            "room": room_name,
            "tick": tick
        }

    def get_rooms_for_tick(self, tick):
        """
        Returns room names that contain dialogue at the given tick.
        Only scans tables with prefix 'RoomTable_'.
        """
        self.cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'RoomTable_%'"
        )
        
        room_tables = [row[0] for row in self.cursor.fetchall()]
        rooms_with_dialogue = []

        for table in room_tables:
            try:
                self.cursor.execute(
                    f"SELECT 1 FROM {table} WHERE tick=? LIMIT 1",
                    (tick,)
                )
                if self.cursor.fetchone():
                    # Extract clean room name
                    room_name = table.replace("RoomTable_", "")
                    rooms_with_dialogue.append(room_name)
            except sqlite3.Error:
                continue

        return rooms_with_dialogue

    def delete_table(self, table_name):
        """
        Deletes a dynamically created table (room or agent log).
        Prevents accidental deletion of core tables.
        """

        # Prevent deleting core system tables
        protected_tables = ["room_desc"]
        if table_name in protected_tables:
            print(f"❌ Cannot delete protected table: {table_name}")
            return False

        try:
            # Check if table exists
            self.cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,)
            )
            if not self.cursor.fetchone():
                print(f"⚠️ Table '{table_name}' does not exist.")
                return False

            # Drop table
            self.cursor.execute(f"DROP TABLE {table_name}")
            self.conn.commit()

            print(f"🗑️ Deleted table: {table_name}")
            return True

        except sqlite3.Error as e:
            print(f"❌ Database Error while deleting table: {e}")
            return False
    def log_official_render(self, tick, room_name):
        """Saves a scene to the 'Director's Cut' timeline."""
        self.cursor.execute(
            "INSERT INTO official_timeline (tick, room_name) VALUES (?, ?)", 
            (tick, room_name)
        )
        self.conn.commit()
        
    def get_official_timeline(self, past_n_ticks):
        """Fetches the actual scripts from the scenes you rendered."""
        self.cursor.execute(
            "SELECT tick, room_name FROM official_timeline ORDER BY tick DESC LIMIT ?", 
            (past_n_ticks,)
        )
        rendered_scenes = self.cursor.fetchall()
        
        # Reverse to chronological order
        rendered_scenes.reverse() 
        
        full_history = []
        for tick, room in rendered_scenes:
            data = self.get_structured_scene_data(tick, room)
            full_history.append(data)
            
        return full_history
    
    def add_to_timeline(self, tick, room_name):
        """Allows the user to explicitly lock a scene into the canon timeline."""
        try:
            self.cursor.execute(
                "INSERT INTO official_timeline (tick, room_name) VALUES (?, ?)", 
                (tick, room_name)
            )
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            print(f"⚠️ Scene {room_name} (Tick {tick}) is already in the timeline.")
            return False

    def get_timeline_history(self, past_n=3):
        """Fetches the last N events added to the timeline in chronological order."""
        self.cursor.execute(
            "SELECT tick, room_name FROM official_timeline ORDER BY id DESC LIMIT ?", 
            (past_n,)
        )
        rows = self.cursor.fetchall()
        rows.reverse() # Flip so the oldest of the 'past N' is first
        
        history_data = []
        for tick, room in rows:
            scene_data = self.get_structured_scene_data(tick, room)
            history_data.append(scene_data)
        return history_data
        
    def close(self):
        self.conn.close()

def generate_gemini_response(sender, receiver, context, room):
    """
    Calls Gemini Pro to generate a dialogue line.
    """
    # 1. Get Persona
    personality = NPC_PERSONAS.get(sender, NPC_PERSONAS["DEFAULT"])
    
    # 2. Build the Prompt
    formatted_prompt = SYSTEM_PROMPT.format(
        sender=sender,
        personality=personality,
        room=room,
        receivers=receiver,
        context=context if context else "(No previous conversation)"
    )

    try:
        # 3. Call the Model
        response = client.models.generate_content(
                model=model, contents=formatted_prompt
            )
        
        # 4. Clean up response (sometimes LLMs add quotes or "Bob: " prefix)
        clean_text = response.text.strip().replace(f"{sender}:", "").replace('"', '')
        return clean_text

    except Exception as e:
        print(f"Gemini API Error: {e}")
        return "...mumbles inaudibly..."

# --- CONTROLLER ---

class JerichoController:
    def __init__(self, game_file_path, db_name="jericho_game.db"):
        self.env = jericho.FrotzEnv(game_file_path)
        self.logger = SQLLogger(db_name)
        self.tick_count = 0
        self.npc_names = ["Bob", "Alice", "Guard"] # Defined for parsing logic
        self.env.reset()
        
    def parse_locations(self, observation_text) -> Dict[str, str]:
        locations = {}
        pattern = re.compile(r"DATA_LOC:\s*(.*?)\s*\|\s*(.*?)(?=DATA_LOC|---|$)")
        for match in pattern.finditer(observation_text):
            locations[match.group(1).strip()] = match.group(2).strip()
        return locations

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


    def update_world_state(self):
        """
        Inspects the Z-Machine memory to get the 'True' state of every room
        and updates the room_desc table.
        """
        # 1. Get all objects
        obs, _, _, _ = self.env.step("dump rooms")
        print(obs)
        # Parse and Store
        self.parse_room_data(obs)

    def conduct_group_chat(self, occupants: List[str], room: str, n_rounds=3):
        print(f"\n--- 🗣️  Group Chat in {room}: {occupants} ---")
        
        for i in range(n_rounds):
            # Simple round-robin: occupants take turns speaking
            speaker = occupants[i % len(occupants)]
            receivers = [o for o in occupants if o != speaker]
            
            # 1. Get Context
            ctx = self.logger.get_agent_context(speaker)
            
            # 2. Generate
            print(f"   [Thinking...] ({speaker})")
            msg = generate_gemini_response(speaker, receivers, ctx, room)
            
            # 3. Broadcast Log
            self.logger.log_broadcast(self.tick_count, room, speaker, receivers, msg)
            print(f"   [{speaker}] to {receivers}: {msg}")
            
            time.sleep(1) # Rate limit

    def step(self):
        self.tick_count += 1
        
        # 1. Advance Game
        obs, _, _, _ = self.env.step('step')
        
        # 2. Update Room Descriptions (New Requirement)
        self.update_world_state()
        
        # 3. Parse Locations & Chat
        locations = self.parse_locations(obs)
        
        # Group Occupants
        room_occupancy = {}
        for npc, room in locations.items():
            if room not in room_occupancy: room_occupancy[room] = []
            room_occupancy[room].append(npc)

        # Trigger Chat if N >= 2
        for room, occupants in room_occupancy.items():
            if len(occupants) >= 2:
                self.conduct_group_chat(occupants, room, n_rounds=3)
        
        return locations

    def run(self):
        print("Starting Simulation with Broadcast & Room Desc...")
        try:
            while True:
                cmd = input(f"\n[Tick {self.tick_count}] Enter to tick, 'q' to quit > ")
                if cmd.lower() == 'q': break
                locs = self.step()
                print(f"📍 Positions: {locs}")
        finally:
            self.logger.close()

class AutomatedDirector:
    def __init__(self):
        self.assets = {} 
        self.model_path = 'selfie_segmenter.tflite'
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

    def create_agent_plate(self, agent_name, agent_desc):
        """Generates a reusable character cutout on a plain background."""
        # We store agents globally, not tied to a specific game, so they persist everywhere
        agent_dir = os.path.join("Assets", "Global_Agents")
        os.makedirs(agent_dir, exist_ok=True)
        filepath = os.path.join(agent_dir, f"{agent_name}.png")

        if os.path.exists(filepath):
            print(f"♻️ Loading existing agent plate for {agent_name}...")
            return filepath

        # Prompt explicitly requests a neutral background for easy removal
        prompt = f"""
        Cinematic mid-shot of {agent_desc}.
        The character is facing the camera.
        Shot in a brightly lit studio with a solid, flat grey background.
        """
        
        print(f"📸 Photographing Agent: {agent_name}...")
        try:
            response = client.models.generate_images(
                model=IMG_MODEL,
                prompt=prompt,
                config=types.GenerateImagesConfig(number_of_images=1, aspect_ratio="1:1") # Square is usually better for character isolation
            )
            response.generated_images[0].image.save(filepath)
            return filepath
        except Exception as e:
            print(f"❌ Failed to photograph {agent_name}: {e}")
            return None

    def create_background_plate(self, room_data, game_name, room_name, facing_direction):
        """Generates the empty background wall."""
        cache_dir = self.get_room_cache_dir(game_name, room_name)
        filepath = os.path.join(cache_dir, f"wall_{facing_direction}.png")

        if os.path.exists(filepath):
            return filepath

        bg_feature = room_data.get(facing_direction, room_data.get('north', 'Empty'))
        
        # Prompt explicitly requests NO people
        prompt = f"""
        Empty background plate, no people, no characters.
        Location style: {room_data['materials']}.
        Visible background: {bg_feature}.
        """
        if room_data.get('is_indoors', True):
            prompt += " Environment: INTERIOR. Enclosed space. Ceiling visible. No sky."

        print(f"🖼️ Painting Background: {room_name} ({facing_direction} wall)...")
        try:
            response = client.models.generate_images(
                model=IMG_MODEL,
                prompt=prompt,
                config=types.GenerateImagesConfig(number_of_images=1, aspect_ratio="16:9")
            )
            response.generated_images[0].image.save(filepath)
            return filepath
        except Exception as e:
            print(f"❌ Failed to paint background: {e}")
            return None

    # --- UPDATED MEDIAPIPE COMPOSITING METHOD ---
    def composite_scene_master(self, agent_path, bg_path, game_name, room_name, agent_name, facing):
        """Cuts the agent out and pastes them onto the background plate using MediaPipe segmentation."""
        cache_dir = self.get_room_cache_dir(game_name, room_name)
        output_path = os.path.join(cache_dir, f"scene_master_{agent_name}_facing_{facing}.png")

        if os.path.exists(output_path):
            return output_path

        print(f"✂️ Compositing {agent_name} into {room_name} via MediaPipe...")
        try:
            # 1. Load the images
            person_img = cv2.imread(agent_path)
            bg_img = cv2.imread(bg_path)

            if person_img is None or bg_img is None:
                print("Error: Could not read images. Check your file paths.")
                return None

            # 2. Setup the MediaPipe Tasks options using the pre-loaded model path
            base_options = python.BaseOptions(model_asset_path=self.model_path)
            options = vision.ImageSegmenterOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                output_category_mask=False,
                output_confidence_masks=True 
            )

            # 3. Initialize the Segmenter and process the image
            with vision.ImageSegmenter.create_from_options(options) as segmenter:
                # Resize person to maintain the 85% cinematic height ratio
                bh, bw, _ = bg_img.shape
                ph, pw, _ = person_img.shape
                
                target_height = int(bh * 0.85) 
                aspect_ratio = pw / ph
                target_width = int(target_height * aspect_ratio)
                
                # Resize person FIRST so the mask matches exactly what we will blend
                person_resized = cv2.resize(person_img, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)

                # Convert BGR (OpenCV) to RGB for MediaPipe
                rgb_person_img = cv2.cvtColor(person_resized, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_person_img)

                # Retrieve the segmentation mask
                segmentation_result = segmenter.segment(mp_image)
                person_mask = segmentation_result.confidence_masks[0].numpy_view()

                # 4. Smooth the mask for a natural look
                person_mask_blurred = cv2.GaussianBlur(person_mask, (7, 7), 0)
                mask_3d = np.stack((person_mask_blurred,) * 3, axis=-1)

                # 5. Composite the images using Alpha Blending (calculating Region of Interest)
                x_offset = (bw - target_width) // 2
                y_offset = bh - target_height

                roi = bg_img[y_offset:y_offset+target_height, x_offset:x_offset+target_width]

                foreground = (person_resized * mask_3d).astype(np.uint8)
                background = (roi * (1 - mask_3d)).astype(np.uint8)
                blended = cv2.add(foreground, background)

                bg_img[y_offset:y_offset+target_height, x_offset:x_offset+target_width] = blended

                # 6. Save the final result
                cv2.imwrite(output_path, bg_img)
                print(f"✅ Success! Saved to {output_path}")
                return output_path
                
        except Exception as e:
            print(f"❌ Compositing failed: {e}")
            
        return None
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

    def film_scene(self, script, scene_assets, game_name, room_name, output_filename):
        """Phase 2: Sends the cached MediaPipe composites to Veo."""
        clips = []
        for i, (speaker, line, emotion) in enumerate(script):
            if speaker not in scene_assets:
                print(f"⚠️ Skipping line for {speaker}: No master asset found.")
                continue
                
            print(f"🎥 Filming Line {i+1}/{len(script)}: {speaker}")
            camera_angle = "Cinematic mid-shot" if i % 2 == 0 else "Cinematic extreme close-up on face"
            with open(scene_assets[speaker], "rb") as f:
                raw_data = f.read()
            
            image_input = types.Image(image_bytes=raw_data, mime_type="image/png")
            text_prompt = f"""
            {camera_angle}.
            The character looks {emotion}.
            The character speaks the following line clearly: "{line}"
            Cinematic lighting, realistic facial animation.
            """
            
            try:
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
                    fname = f"clip_{i:03d}_{speaker}.mp4"
                    downloaded = self._download_video(operation.result.generated_videos[0].video.uri, fname)
                    if downloaded:
                        clips.append(downloaded)
            except Exception as e:
                print(f"❌ Clip failed: {e}")

        # Stitch
        if clips:
            print(f"🎞️ Stitching {len(clips)} clips...")
            with open(f"ffmpeg_list_{room_name}.txt", "w") as f:
                for vid in clips:
                    f.write(f"file '{vid}'\n")
            
            subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", f"ffmpeg_list_{room_name}.txt", "-c", "copy", output_filename], check=True)
            os.remove(f"ffmpeg_list_{room_name}.txt")
            
            for vid in clips:
                if os.path.exists(vid): os.remove(vid)
                    
            print(f"🍿 Scene Complete: {output_filename}")
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
        print("🎬 Filming Recap Video...")
        return self.film_scene(script, scene_assets, game_name, "Recap_Studio", output_filename)

class SceneSelector:

    def __init__(self, db, director, key_terms=None,game_name="UnknownGame"):
        self.db = db
        self.director = director
        self.room_asset_cache = {}
        self.game_name = game_name 
    
        # INTEGRATION: Initialize the Drama Engine
        # We pass your specific game terms here
        self.scorer = DramaScorer(key_terms=key_terms) 

    def scan_and_rank(self, start_tick, end_tick, top_n=5):
        """
        Scans a range of ticks, scores every conversation, 
        and returns the top N 'Must Watch' moments.
        """
        print(f"🕵️  Scanning ticks {start_tick} to {end_tick} for drama...")
        
        ranked_scenes = [] # Will store tuples: (-score, tick, room, data)

        for tick in range(start_tick, end_tick + 1):
            rooms = self.db.get_rooms_for_tick(tick)
            if not rooms:
                continue

            for room in rooms:
                scene_data = self.db.get_structured_scene_data(tick, room)
                script_data = scene_data.get("script", [])

                if not script_data or len(script_data) < 2:
                    continue  # Skip empty or single-line events

                # ADAPTER STEP: Convert DB format to Scorer format
                # Your DB uses "line", Scorer uses "text"
                scorer_input = [
                    {"speaker": row["speaker"], "text": row["line"]} 
                    for row in script_data
                ]

                # CALCULATE SCORE
                score = self.scorer.calculate_score(scorer_input)

                # Store if it has any meaningful content (> 10)
                if score > 10:
                    # Python's heap sort is min-heap, so we store negative score to get max
                    heapq.heappush(ranked_scenes, (-score, tick, room, scene_data))

        # Output the results
        print(f"\n🎬 TOP {top_n} DRAMATIC MOMENTS FOUND:")
        
        # Extract top N from heap
        top_candidates = []
        while ranked_scenes and len(top_candidates) < top_n:
            neg_score, tick, room, data = heapq.heappop(ranked_scenes)
            real_score = -neg_score
            top_candidates.append((real_score, tick, room, data))
            
            # Print a preview for the user
            snippet = data['script'][0]['line'][:50] + "..."
            print(f"[{len(top_candidates)}] Score: {real_score:.1f} | Tick: {tick} | Room: {room}")
            print(f"    Preview: \"{snippet}\"")

        return top_candidates

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
            script.append((speaker, text, emotion))
            
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