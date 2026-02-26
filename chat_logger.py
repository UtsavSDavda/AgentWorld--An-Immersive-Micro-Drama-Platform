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
import re
from textblob import TextBlob
import json
from nrclex import NRCLex
import csv 

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

    def _design_set_automatically(self, raw_room_desc):
        """
        Uses LLM to expand a single game description into a full 3D room layout.
        """
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
        raw_text = response.text
        print(f"DEBUG: Raw LLM Output:\n{raw_text}") 
        data = self._clean_and_parse_json(raw_text)
        if not data:
            # Fallback default if parsing fails completely
            print("❌ Parsing failed. Using default empty room.")
            return {
                "materials": "Generic walls",
                "north": "Empty", "south": "Empty", "east": "Empty", "west": "Empty",
                "is_indoors": True
            }
        if isinstance(data, list):
            if len(data) > 0:
                data = data[0]
            else:
                print("❌ Empty list returned. Using fallback.")
                return {
                    "materials": "Generic walls",
                    "north": "Empty", "south": "Empty",
                    "east": "Empty", "west": "Empty",
                    "is_indoors": True
                }

        return data

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

    def create_master_assets(self, agents, room_data):
        """
        Generates consistent images for all agents based on the Room Geometry.
        agents = [{'name': 'Guard', 'desc': '...', 'facing': 'north'}]
        """
        for agent in agents:
            name = agent['name']
            desc = agent['desc']
            facing = agent.get('facing', 'north').lower()
            
            # 1. Determine Background based on facing direction
            # If I look North, I see the North wall.
            bg_feature = room_data.get(facing, room_data['north'])
            
            prompt = f"""
            Cinematic mid-shot of {desc}.
            The character is facing the camera (body oriented {facing}).
            
            BACKGROUND:
            Location style: {room_data['materials']}.
            Visible background: {bg_feature}.
            """
            
            if room_data['is_indoors']:
                prompt += " Environment: INTERIOR. Enclosed space. Ceiling visible. No sky."

            print(f"📸 Casting {name} (Facing {facing})...")
            
            try:
                response = client.models.generate_images(
                    model=IMG_MODEL,
                    prompt=prompt,
                    config=types.GenerateImagesConfig(number_of_images=1, aspect_ratio="16:9")
                )
                filename = f"master_{name}.png"
                response.generated_images[0].image.save(filename)
                self.assets[name] = filename
            except Exception as e:
                print(f"❌ Failed to cast {name}: {e}")

    def produce_scene(self, room_description, agents, script, output_filename="jericho_scene.mp4"):
        """
        The Main Function: Takes raw game data -> Final MP4
        """
        # Step 1: Design the Set (Auto-Hallucinate 4 walls)
        room_data = self._design_set_automatically(room_description)
        
        # Step 2: Generate Master Assets
        self.create_master_assets(agents, room_data)
        
        # Step 3: Film the Script
        clips = []
        for i, (speaker, line, emotion) in enumerate(script):
            if speaker not in self.assets:
                print(f"⚠️ Skipping line for unknown speaker: {speaker}")
                continue
                
            print(f"🎥 Filming Line {i+1}/{len(script)}: {speaker}")
            
            # Load Asset
            with open(self.assets[speaker], "rb") as f:
                raw_data = f.read()
            
            image_input = types.Image(image_bytes=raw_data, mime_type="image/png")
            text_prompt = f"""
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
                
                while not operation.done:
                    time.sleep(3)
                    operation = client.operations.get(operation) # Fixed polling

                if operation.result:
                    fname = f"clip_{i:03d}_{speaker}.mp4"
                    # Fixed Download Logic
                    downloaded = self._download_video(operation.result.generated_videos[0].video.uri, fname)
                    if downloaded:
                        clips.append(downloaded)
            except Exception as e:
                print(f"❌ Clip failed: {e}")

        # Step 4: Stitch
        if clips:
            print(f"🎞️ Stitching {len(clips)} clips...")
            with open("ffmpeg_list.txt", "w") as f:
                for vid in clips:
                    f.write(f"file '{vid}'\n")
            
            subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "ffmpeg_list.txt", "-c", "copy", output_filename], check=True)
            os.remove("ffmpeg_list.txt")
            print(f"🍿 Scene Complete: {output_filename}")
        else:
            print("⚠️ No clips generated.")

class SceneSelector:

    def __init__(self, db, director, key_terms=None):
        self.db = db
        self.director = director
        self.room_asset_cache = {}
        
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
        # (This remains largely the same, but now we can inject the score if we want)
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
            
            # IMPROVEMENT: Use TextBlob again here for line-by-line emotion?
            # For now, we keep it neutral or let the Director handle it.
            emotion = self._detect_emotion_nrc(text)
            print(emotion)
            script.append((speaker, text, emotion))
            
            if speaker not in speakers:
                agents.append({
                    "name": speaker,
                    "desc": f"{speaker}, a character in the Jericho world",
                    "facing": "north"
                })
                speakers.add(speaker)

        if room_name not in self.room_asset_cache:
            print(f"🆕 Designing new master assets for room: {room_name}")
            # Assuming your director has this method
            room_data = self.director._design_set_automatically(room_visual)
            self.director.create_master_assets(agents, room_data)
            self.room_asset_cache[room_name] = True
        else:
            print(f"♻️ Reusing master assets for room: {room_name}")

        print(f"🎥 Rendering Tick {scene_data.get('tick')}...")
        self.director.produce_scene(
            room_description=room_visual,
            agents=agents,
            script=script,
            output_filename=f"scene_tick_{scene_data.get('tick', 0)}_{room_name}.mp4"
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