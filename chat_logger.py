import jericho
import re
import sqlite3
from typing import Dict, List
from google import genai
import os
from dotenv import load_dotenv
from videoprompts import SYSTEM_PROMPT

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI2")

client = genai.Client(api_key=GEMINI_API_KEY)
model = "gemini-2.0-flash"

NPC_PERSONAS = {
    "Bob": "You are Bob, a paranoid scientist. You believe the lab is compromised. You speak in short, nervous sentences.",
    "Alice": "You are Alice, a cheerful but clumsy intern. You are trying to hide the fact that you broke a beaker earlier.",
    "DEFAULT": "You are an inhabitant of this game world. You are curious and conversational."
}
# --- ADVANCED SQL LOGGER ---

# class SQLLogger:
#     def __init__(self, db_name="jericho_game.db"):
#         self.conn = sqlite3.connect(db_name)
#         self.cursor = self.conn.cursor()
#         self.agent_memory_cache = {} # Quick access for LLM context

#     def sanitize_name(self, name):
#         """Converts 'Living Room' -> 'Table_Living_Room' or 'Bob' -> 'Table_Bob'"""
#         clean_name = re.sub(r'\W+', '_', name)
#         return f"Table_{clean_name}"

#     def ensure_table(self, table_name, is_room_log=False):
#         """
#         Creates a table dynamically. 
#         Room Logs don't need 'Room' column (it's implicit).
#         NPC Logs DO need 'Room' column to know where they were.
#         """
#         if is_room_log:
#             # Schema for ROOMS: Tick, Sender, Receiver, Message
#             sql = f"""CREATE TABLE IF NOT EXISTS {table_name} (
#                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                 tick INTEGER,
#                 sender TEXT,
#                 receiver TEXT,
#                 message TEXT
#             );"""
#         else:
#             # Schema for NPCs: Tick, Room, Sender, Receiver, Message
#             sql = f"""CREATE TABLE IF NOT EXISTS {table_name} (
#                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                 tick INTEGER,
#                 room TEXT,
#                 sender TEXT,
#                 receiver TEXT,
#                 message TEXT
#             );"""
            
#         self.cursor.execute(sql)
#         self.conn.commit()

#     def log_chat(self, tick, room, sender, receiver, message):
#         # 1. LOG TO ROOM TABLE
#         room_table = self.sanitize_name(room)
#         self.ensure_table(room_table, is_room_log=True)
#         self.cursor.execute(
#             f"INSERT INTO {room_table} (tick, sender, receiver, message) VALUES (?, ?, ?, ?)",
#             (tick, sender, receiver, message)
#         )

#         # 2. LOG TO SENDER'S TABLE (Personal History)
#         sender_table = self.sanitize_name(sender)
#         self.ensure_table(sender_table, is_room_log=False)
#         self.cursor.execute(
#             f"INSERT INTO {sender_table} (tick, room, sender, receiver, message) VALUES (?, ?, ?, ?, ?)",
#             (tick, room, sender, receiver, message)
#         )

#         # 3. LOG TO RECEIVER'S TABLE (Personal Inbox)
#         receiver_table = self.sanitize_name(receiver)
#         self.ensure_table(receiver_table, is_room_log=False)
#         self.cursor.execute(
#             f"INSERT INTO {receiver_table} (tick, room, sender, receiver, message) VALUES (?, ?, ?, ?, ?)",
#             (tick, room, sender, receiver, message)
#         )

#         self.conn.commit()
        
#         # 4. Update in-memory cache for fast LLM context retrieval
#         self.update_cache(sender, f"You said to {receiver}: {message}")
#         self.update_cache(receiver, f"{sender} said to you: {message}")

#     def update_cache(self, agent, text):
#         if agent not in self.agent_memory_cache:
#             self.agent_memory_cache[agent] = []
#         self.agent_memory_cache[agent].append(text)

#     def get_agent_context(self, agent, limit=3):
#         # Option A: Read from Memory Cache (Faster)
#         # history = self.agent_memory_cache.get(agent, [])
#         # return "\n".join(history[-limit:])
    
#         #Option B: Read from SQL (Slower but persistent across restarts)
#         table = self.sanitize_name(agent)
#         self.cursor.execute(f"SELECT sender, message FROM {table} ORDER BY id DESC LIMIT ?", (limit,))
#         # ...

#     def close(self):
#         self.conn.close()
class SQLLogger:
    def __init__(self, db_name="jericho_game.db"):
        self.db_name = db_name
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self.cursor = self.conn.cursor()

    def sanitize_name(self, name):
        """Converts 'Bob' -> 'Table_Bob' to be SQL safe."""
        clean_name = re.sub(r'\W+', '_', name)
        return f"Table_{clean_name}"

    def ensure_table(self, table_name, is_room_log=False):
        if is_room_log:
            # Schema for ROOMS
            sql = f"""CREATE TABLE IF NOT EXISTS {table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tick INTEGER,
                sender TEXT,
                receiver TEXT,
                message TEXT
            );"""
        else:
            # Schema for NPCs
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

    def log_chat(self, tick, room, sender, receiver, message):
        try:
            # 1. LOG TO ROOM TABLE
            room_table = self.sanitize_name(room)
            self.ensure_table(room_table, is_room_log=True)
            self.cursor.execute(
                f"INSERT INTO {room_table} (tick, sender, receiver, message) VALUES (?, ?, ?, ?)",
                (tick, sender, receiver, message)
            )

            # 2. LOG TO SENDER'S TABLE
            sender_table = self.sanitize_name(sender)
            self.ensure_table(sender_table, is_room_log=False)
            self.cursor.execute(
                f"INSERT INTO {sender_table} (tick, room, sender, receiver, message) VALUES (?, ?, ?, ?, ?)",
                (tick, room, sender, receiver, message)
            )

            # 3. LOG TO RECEIVER'S TABLE
            receiver_table = self.sanitize_name(receiver)
            self.ensure_table(receiver_table, is_room_log=False)
            self.cursor.execute(
                f"INSERT INTO {receiver_table} (tick, room, sender, receiver, message) VALUES (?, ?, ?, ?, ?)",
                (tick, room, sender, receiver, message)
            )

            self.conn.commit()
            
        except sqlite3.Error as e:
            print(f"❌ Database Error during Write: {e}")

    def get_agent_context(self, agent_name, limit=5):
        """
        Retrieves the last N messages from the DB for a specific agent.
        """
        table_name = self.sanitize_name(agent_name)
        
        try:
            # Check if table exists first to avoid crashing on new agents
            self.cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?", 
                (table_name,)
            )
            if not self.cursor.fetchone():
                return ""  # Table doesn't exist yet, so no context.

            # Retrieve last N rows
            query = f"""
                SELECT sender, message, tick 
                FROM {table_name} 
                ORDER BY id DESC 
                LIMIT ?
            """
            self.cursor.execute(query, (limit,))
            rows = self.cursor.fetchall()
            
            # The DB returns them in DESC order (Newest -> Oldest).
            # We must reverse them to readable chronological order (Oldest -> Newest).
            rows.reverse()
            
            # Format nicely for the LLM
            context_lines = []
            for row in rows:
                sender, msg, tick = row
                # We add the sender name so the LLM knows who said what
                context_lines.append(f"[Tick {tick}] {sender}: {msg}")
            
            return "\n".join(context_lines)

        except sqlite3.Error as e:
            print(f"⚠️ Could not retrieve context from DB: {e}")
            return ""

    def close(self):
        self.conn.close()
# --- MOCK LLM (Replace with Gemini) ---

def generate_reply(sender, receiver, context, room):
    return f"Hello {receiver}, I see you are also in the {room}."

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
        receiver=receiver,
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

# --- GAME CONTROLLER ---

class JerichoController:
    def __init__(self, game_file_path):
        print(f"Loading Game: {game_file_path}...")
        self.env = jericho.FrotzEnv(game_file_path)
        self.logger = SQLLogger("jericho_game.db")
        self.tick_count = 0
        self.env.reset()
        
    def parse_locations(self, observation_text) -> Dict[str, str]:
        locations = {}
        # Robust regex for finding DATA_LOC tags
        pattern = re.compile(r"DATA_LOC:\s*(.*?)\s*\|\s*(.*?)(?=DATA_LOC|---|$)")
        
        for match in pattern.finditer(observation_text):
            name = match.group(1).strip()
            room = match.group(2).strip()
            locations[name] = room
        return locations

    def check_collisions_and_chat(self, locations):
        room_occupancy = {}
        for npc, room in locations.items():
            if room not in room_occupancy: room_occupancy[room] = []
            room_occupancy[room].append(npc)

        for room, occupants in room_occupancy.items():
            if len(occupants) >= 2:
                self.chat_n_times(occupants[0], occupants[1], room, n=3)

    def chat_n_times(self, agent_a, agent_b, room, n=3):
        print(f"\n--- 🗣️  Interaction: {agent_a} & {agent_b} in {room} ---")
        
        for _ in range(n):
            # A talks
            ctx_a = self.logger.get_agent_context(agent_a)
            msg_a = generate_gemini_response(agent_a, agent_b, ctx_a, room)
            self.logger.log_chat(self.tick_count, room, agent_a, agent_b, msg_a)
            print(f"   [{agent_a}]: {msg_a}")

            # B talks
            ctx_b = self.logger.get_agent_context(agent_b)
            msg_b = generate_gemini_response(agent_b, agent_a, ctx_b, room)
            self.logger.log_chat(self.tick_count, room, agent_b, agent_a, msg_b)
            print(f"   [{agent_b}]: {msg_b}")

    def step(self):
        self.tick_count += 1
        observation_text, reward, done, info = self.env.step('step')
        locations = self.parse_locations(observation_text)
        
        if locations:
            self.check_collisions_and_chat(locations)
        
        return locations

    def run(self):
        print("Starting Simulation with Dual SQL Logging (Room + NPC)...")
        try:
            while True:
                cmd = input(f"\n[Tick {self.tick_count}] Enter to tick, 'q' to quit > ")
                if cmd.lower() == 'q': break
                
                locs = self.step()
                print(f"📍 Positions: {locs}")
        finally:
            self.logger.close()
            print("Database connection closed.")

# --- EXECUTION ---
if __name__ == "__main__":
    GAME_FILE = "Control3.z8"  
    
    try:
        # print(generate_gemini_response(sender="Bob",receiver="Alice",context="Bob is just rich!",room="Hallway"))
        controller = JerichoController(GAME_FILE)
        controller.run()
    except FileNotFoundError:
        print(f"Error: Could not find game file '{GAME_FILE}'")