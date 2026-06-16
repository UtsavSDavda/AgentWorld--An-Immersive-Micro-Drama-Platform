import jericho
import re
import time
import os
from google import genai
from dotenv import load_dotenv
import json
from google.genai import types
# Import your existing database logger and chat generation functions
from chat_logger import SQLLogger, generate_gemini_response, GameDBManager

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_ID = "gemini-2.5-flash"

class BaselineAgent:
    def __init__(self, name, db_logger):
        self.name = name
        self.db = db_logger
        self.current_room = "Unknown"
        
        # 1. Fetch Persona from Database
        profile = self.db.get_npc_profile(self.name)
        self.base_persona = profile[0] if profile else f"You are {self.name}."
        
        # 2. Store the temporary psychological shifts
        self.active_deltas = []

    @property
    def full_persona(self):
        """Dynamically combines the core identity with current emotional states."""
        if not self.active_deltas:
            return self.base_persona
        # Append the accumulated deltas to the base persona
        return f"{self.base_persona} CURRENT STATE OF MIND: {' '.join(self.active_deltas)}"

    def reflect(self):
        """The Impact Scoring Method: Evaluates recent memory to generate Persona Deltas."""
        # Fetch the last 5 chat messages/events from the database
        chat_memory = self.db.get_agent_context(self.name, limit=15)
        
        # If nothing has happened, skip reflection to save API calls
        if not chat_memory:
            return

        prompt = f"""
        You are {self.name}.
        Your core persona is: {self.base_persona}
        
        RECENT EVENTS AND CONVERSATIONS:
        {chat_memory}
        
        TASK:
        1. Rate the 'Impact Level' of these recent events on your psyche from 1 (trivial/normal) to 10 (life-altering/traumatic).
        2. If the impact is 6 or higher, write a 1-sentence 'Persona Delta' describing how your attitude, mood, or belief has just changed. If the impact is 5 or below, leave the delta entirely blank.
        
        Respond ONLY with valid JSON in this exact format:
        {{
            "impact_score": 5,
            "persona_delta": ""
        }}
        """
        
        try:
            # Force Gemini to output structured JSON
            response = client.models.generate_content(
                model=MODEL_ID, 
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            
            # Parse the LLM's decision
            data = json.loads(response.text)
            score = data.get("impact_score", 0)
            delta = data.get("persona_delta", "").strip()
            
            # The Heuristic: Decide WHETHER to add, and WHAT to add
            if score >= 6 and delta:
                print(f"  [💡 PSYCHOLOGICAL SHIFT] {self.name} experienced a high-impact event (Score: {score}/10)!")
                print(f"      Delta Added: '{delta}'")
                
                self.active_deltas.append(delta)
                
                # Safeguard: Cap the deltas at 2 so the agent doesn't become schizophrenic
                if len(self.active_deltas) > 2:
                    self.active_deltas.pop(0) # Remove the oldest mood
            else:
                print("No shift in persona this turn. Impact Score:", score)
                pass # Low impact; no persona change required

        except Exception as e:
            print(f"  [!] Reflection failed for {self.name}: {e}")

    def generate_action_catalog(self, context_data):
        """
        UNRESTRICTED MODE: Offers logically opposing actions (open/close, on/off) 
        to allow the LLM to make mistakes and trigger Inform FAIL states.
        """
        catalog = ["wait"]
        if not context_data:
            return catalog

        # 1. Movement
        for direction in context_data.get("direction", []):
            catalog.append(f"go {direction}")
            
        # 2. Social Interactions
        for person in context_data.get("person", []):
            catalog.append(f"examine {person}")
            catalog.append(f"talk to {person}") 
            
        # 3. Environment Objects
        for obj_data in context_data.get("object", []):
            name = obj_data["name"]
            state = obj_data["state"] # e.g., 'device_off', 'openable_closed'
            portability = obj_data["portability"]
            lock_status = obj_data.get("lock", "unlocked") # <-- Fetch the lock status
            
            catalog.append(f"examine {name}")
            
            # Physics Filter: Portability
            if portability == "portable":
                catalog.append(f"take {name}")
                
            # Unrestricted Mechanics: Devices
            if "device" in state:
                catalog.append(f"switch on {name}")
                catalog.append(f"switch off {name}")
                
            # Unrestricted Mechanics: Openable Containers/Doors
            if "openable" in state:
                catalog.append(f"open {name}")
                catalog.append(f"close {name}")
                catalog.append(f"look inside {name}")
                
            # <-- NEW: Unrestricted Mechanics for Locks -->
            if lock_status == "locked":
                catalog.append(f"unlock {name}")
                
        # 4. Inventory Actions
        for inv_data in context_data.get("inventory", []):
            name = inv_data["name"]
            state = inv_data["state"]
            
            catalog.append(f"examine {name}")
            catalog.append(f"drop {name}")
            
            # Unrestricted Mechanics for objects the agent is holding
            if "device" in state: 
                catalog.append(f"switch on {name}")
                catalog.append(f"switch off {name}")
                
            if "openable" in state:
                catalog.append(f"open {name}")
                catalog.append(f"close {name}")
                catalog.append(f"look inside {name}")

        return sorted(list(set(catalog)))

    def decide_action(self, context_data, debug_mode=True):
        """Prompts Gemini using the dynamic full_persona and Chat Memory."""
        catalog = self.generate_action_catalog(context_data)
        
        # --- DEBUG: Print the generated catalog ---
        print(f"\n⚙️ [DEBUG] Generated Action Catalog for {self.name}:")
        for action in catalog:
            if any(keyword in action for keyword in ["open", "close", "switch on", "switch off"]):
                print(f"    -> {action}  <-- [Opposing Option Available]")
            else:
                print(f"    -> {action}")
        
        # Fetch Chat Memory
        chat_memory = self.db.get_agent_context(self.name, limit=15)
        
        visible_items = [obj["name"] for obj in context_data.get("object", [])]
        visible_people = context_data.get("person", [])
        
        prompt = f"""
        You are {self.name}. 
        Persona: {self.full_persona}
        
        CHAT MEMORY (Recent conversations you heard or participated in):
        {chat_memory if chat_memory else "No recent conversations."}
        
        CURRENT ENVIRONMENT:
        You are in: {self.current_room}
        People here: {', '.join(visible_people) if visible_people else 'No one'}
        Objects here: {', '.join(visible_items) if visible_items else 'Nothing'}
        
        TASK:
        Based on your persona and recent conversations, choose EXACTLY ONE action from the list below. 
        Output ONLY the exact string of the action. Do not add quotes, punctuation, or explanations.
        
        AVAILABLE ACTIONS:
        {chr(10).join(catalog)}
        """

        if debug_mode:
            print(f"\n[DEBUG: {self.name}'s Prompt] {'-'*40}")
            print(prompt.strip())
            print(f"[{'-'*60}]")

        try:
            response = client.models.generate_content(model=MODEL_ID, contents=prompt)
            chosen_action = response.text.strip()
            
            # --- DEBUG: Print the exact selection ---
            print(f"🧠 [DEBUG] {self.name} selected: '{chosen_action}'\n")
            
            if chosen_action not in catalog:
                print(f"  [!] {self.name} hallucinated '{chosen_action}'. Defaulting to 'wait'.")
                return "wait"
            return chosen_action
            
        except Exception as e:
            print(f"  [!] API Error for {self.name}: {e}. Defaulting to 'wait'.")
            return "wait"


class BaselineRunner:
    def __init__(self, game_file, session_id):
        print(f"Loading game environment: {game_file}...")
        self.env = jericho.FrotzEnv(game_file)
        self.tick_count = 0
        self.agents = {} 
        
        # Initialize Database Logger
        db_manager = GameDBManager()
        db_name = db_manager.get_or_create_user_db(game_file, session_id)
        self.db = SQLLogger(session_id)
        
        self._initialize_agents_from_inform()

    def _initialize_agents_from_inform(self):
        """Extracts agents from Inform and registers them with the DB and Runner."""
        obs, _, _, _ = self.env.step("dump agents")
        pattern = re.compile(r"DATA_NPC:\s*(.*?)\s*\|\s*(.*?)(?=DATA_NPC|--- END NPC DATA ---|$)", re.DOTALL)
        
        print("\n--- Initializing AI Agents ---")
        for match in pattern.finditer(obs):
            name = match.group(1).strip()
            desc = match.group(2).strip()
            
            # Save to DB (creates profile if missing)
            if not self.db.get_npc_profile(name):
                # Simple default persona creation for the baseline
                self.db.save_npc_profile(
                    name=name, raw_desc=desc, persona=f"You are {name}. {desc}", 
                    appearance=f"A portrait of {name}", gender="NEUTRAL", voice_id="en-US-Neural2-J"
                )
                
            self.agents[name] = BaselineAgent(name, self.db)
            print(f"Loaded {name} into system.")
    
    def naturalize_action(self, name, raw_action):
        """
        Converts Inform 7 Standard Rule gerunds into natural script stage directions.
        Handles both built-in actions and dynamically conjugates custom actions.
        """
        # Slice out the character's name if Inform prepended it
        action_text = raw_action[len(name):].strip() if raw_action.startswith(name) else raw_action.strip()
        
        # Split the first word (the gerund) from the rest of the action string
        parts = action_text.split(" ", 1)
        gerund = parts[0].lower()
        remainder = " " + parts[1] if len(parts) > 1 else ""

        # Comprehensive Inform 7 Standard Actions Dictionary
        # Maps the root gerund to its third-person present tense
        gerund_map = {
            # Movement & Positioning
            "going": "goes",
            "entering": "enters",
            "exiting": "exits",
            "getting": "gets",       # Handles "getting off"
            "jumping": "jumps",
            "climbing": "climbs",
            
            # Inventory & Manipulation
            "taking": "takes",       # Handles "taking off"
            "dropping": "drops",
            "putting": "puts",       # Handles "putting on"
            "inserting": "inserts",  # Handles "inserting into"
            "wearing": "wears",
            "removing": "removes",   
            
            # Perception & Investigation
            "examining": "examines",
            "looking": "looks",      # Handles "looking under"
            "searching": "searches",
            "listening": "listens",  # Handles "listening to"
            "smelling": "smells",
            "tasting": "tastes",
            "touching": "touches",
            
            # Mechanical & Device Interaction
            "opening": "opens",
            "closing": "closes",
            "locking": "locks",
            "unlocking": "unlocks",
            "switching": "switches", # Handles "switching on/off"
            "pushing": "pushes",
            "pulling": "pulls",
            "turning": "turns",
            "tying": "ties",         # Handles "tying it to"
            "setting": "sets",
            
            # Social & Combat
            "asking": "asks",        # Handles "asking it about"
            "telling": "tells",
            "answering": "answers",
            "saying": "says",        # Handles "saying yes/sorry"
            "showing": "shows",
            "giving": "gives",
            "attacking": "attacks",
            "kissing": "kisses",
            "waking": "wakes",       # Handles "waking up"
            
            # Miscellaneous
            "waiting": "waits",
            "eating": "eats",
            "drinking": "drinks",
            "throwing": "throws",    # Handles "throwing it at"
            "rubbing": "rubs",
            "waving": "waves",
            "burning": "burns",
            "cutting": "cuts",
            "squeezing": "squeezes",
            "swinging": "swings",
            "sleeping": "sleeps"
        }

        # 1. Check if it is a built-in Inform 7 action
        if gerund in gerund_map:
            return f"{name} {gerund_map[gerund]}{remainder}."
        
        # 2. Algorithmic fallback for custom actions (e.g., "untrapping" -> "untraps")
        if gerund.endswith("ing"):
            base = gerund[:-3] # Remove 'ing'
            
            # Handle English spelling rules for third-person singular
            if base.endswith("y"):
                base = base[:-1] + "ies"
            elif base.endswith(("s", "sh", "ch", "x", "z")):
                base += "es"
            # Handle double consonants (e.g., "hugging" -> "hugg" -> "hugs")
            elif len(base) > 2 and base[-1] == base[-2]:
                 base = base[:-1] + "s"
            else:
                base += "s"
            return f"{name} {base}{remainder}."

        # 3. Ultimate fail-safe
        return f"{name} is {action_text}."

    def parse_sight_data(self, observation_text, tick_count):
        """
        Parses the 6-part DATA_SIGHT logs from the upgraded Inform 7 engine.
        Extracts physical constraints (state, portability, lock) alongside the object name.
        """
        sight_dict = {}
        for line in observation_text.splitlines():
            line = line.strip()
            if line.startswith("DATA_SIGHT:"):
                # Log the raw data to the database as ground truth
                self.db.log_engine_data(tick_count, "DATA_SIGHT", line)
                
                clean_line = line.replace("DATA_SIGHT:", "").strip()
                parts = [p.strip() for p in clean_line.split("|")]
                
                if len(parts) >= 3:
                    observer, category, target = parts[0], parts[1], parts[2]
                    
                    if observer not in sight_dict:
                        sight_dict[observer] = {"direction": [], "person": [], "object": [], "inventory": []}
                    
                    if category in ["object", "inventory"]:
                        # Safely unpack the new 6-part physics data with fallbacks
                        state = parts[3] if len(parts) >= 4 else "basic"
                        portability = parts[4] if len(parts) >= 5 else "portable" 
                        lock_status = parts[5] if len(parts) >= 6 else "unlocked"
                        
                        sight_dict[observer][category].append({
                            "name": target,
                            "state": state,
                            "portability": portability,
                            "lock": lock_status
                        })
                    else:
                        sight_dict[observer][category].append(target)
                        
        return sight_dict

    def parse_and_print_locations(self, observation_text,tick_count):
        locations = {}
        print("\n📍 CURRENT LOCATIONS:")
        for line in observation_text.splitlines():
            line = line.strip()
            if line.startswith("DATA_LOC:"):
                self.db.log_engine_data(tick_count, "DATA_LOC", line)
                clean_line = line.replace("DATA_LOC:", "").strip()
                parts = [p.strip() for p in clean_line.split("|")]
                if len(parts) == 2:
                    name, room = parts[0], parts[1]
                    locations[name] = room
                    print(f"  - {name} is in the {room}")
                    if name in self.agents:
                        self.agents[name].current_room = room
        return locations

    def filter_narrative(self, observation_text):
        clean_lines = []
        skip_blocks = [
            "--- START DATA ---", "--- END DATA ---", 
            "--- START SIGHT ---", "--- END SIGHT ---", 
            "--- START ROOM DATA ---", "--- END ROOM DATA ---",
            "--- BEGIN NPC DATA ---", "--- END NPC DATA ---",
            "--- Simulation Step ---", 
            "DATA_RESULT", "DATA_EVENT", "DATA_LOC", 
            "DATA_SIGHT", "DATA_NPC", "DATA_ROOM"
        ]
        
        for line in observation_text.splitlines():
            clean_line = line.strip()
            
            if not clean_line:
                continue
                
            # Catch the Inform 7 prompt, even if it has trailing spaces
            if clean_line.startswith(">"):
                continue
                
            # Failsafe: Catch any leaking "executes plan" lines just in case
            if "executes plan:" in clean_line:
                continue
                
            # Skip standalone room names that Inform tries to print on refresh
            if clean_line in ["Security Kiosk", "Server Corridor", "Data Vault", "Unknown"]:
                continue
                
            if any(clean_line.startswith(block) for block in skip_blocks):
                continue
                
            clean_lines.append(clean_line)
            
        return "\n".join(clean_lines)
    
    def parse_physical_outcomes(self, observation_text, tick_count):
        """Parses DATA_RESULT output to generate verified stage directions."""
        for line in observation_text.splitlines():
            line = line.strip()
            if line.startswith("DATA_RESULT:"):
                self.db.log_engine_data(tick_count, "DATA_RESULT", line)
                clean = line.replace("DATA_RESULT:", "").strip()
                parts = [p.strip() for p in clean.split("|")]
                
                if len(parts) >= 3:
                    name = parts[0]
                    status = parts[1]
                    raw_action = parts[2] 
                    
                    room = self.agents[name].current_room if name in self.agents else "Unknown"
                    
                    if status == "SUCCESS":
                        # Replaces the robotic string with natural narrative
                        stage_direction = self.naturalize_action(name, raw_action)
                    else:
                        reason = parts[3] if len(parts) > 3 else "blocked"
                        # Clean up the text for failures to maintain readability
                        clean_action = raw_action[len(name):].strip() if raw_action.startswith(name) else raw_action.strip()
                        stage_direction = f"{name} tries to {clean_action}, but fails ({reason})."
                    
                    self.db.log_broadcast(tick_count, room, "ACTION", ["ALL"], stage_direction)

    def run(self, max_ticks=50):
        import time
        
        print("\n=== STARTING BASELINE SCENARIO ===")
        # Force the game to print locations on turn 0
        init_obs, _, _, _ = self.env.step("wait") 
        self.parse_and_print_locations(init_obs, self.tick_count) 
        obs, _, _, _ = self.env.step("dump sight")
        
        # Parse Tick 0's sight data BEFORE the loop starts
        current_sight_data = self.parse_sight_data(obs, self.tick_count)
        
        # --- NEW: Initialize the fast-forward counter ---
        auto_ticks = 0 
        
        while self.tick_count < max_ticks:
            self.tick_count += 1
            print(f"\n{'='*20} TICK {self.tick_count} {'='*20}")
            
            # --- REFLECTION LOOP ---
            if self.tick_count % 3 == 0:
                print("\n🧘 Agents are reflecting on recent events...")
                for _, agent in self.agents.items():
                    agent.reflect()
            
            print("\n🧠 Agents are making decisions...")
            
            # Track who is examining what this tick
            examine_targets = {}
            
            # --- BEAT 1: INTENT & DIALOGUE ---
            for agent_name, agent in self.agents.items():
                agent_context = current_sight_data.get(agent_name, {})
                chosen_action = agent.decide_action(agent_context, debug_mode=False)
                
                if chosen_action.startswith("talk to "):
                    target_person = chosen_action.replace("talk to ", "").strip()
                    chat_history = self.db.get_agent_context(agent_name)
                    
                    dialogue_line = generate_gemini_response(
                        sender=agent_name, receiver=[target_person], 
                        context=chat_history, room=agent.current_room, db_logger=self.db
                    )
                    
                    if dialogue_line:
                        self.db.log_broadcast(self.tick_count, agent.current_room, agent_name, [target_person], dialogue_line)
                        self.db.add_to_timeline(self.tick_count, agent.current_room)
                        print(f"  [DIALOGUE] {agent_name}: \"{dialogue_line}\"")
                    
                    self.env.step(f"force {agent_name} to wait")
                else:
                    self.env.step(f"force {agent_name} to {chosen_action}")
                    
                    # ---> LOG EXAMINE INTENT <---
                    if chosen_action.startswith("examine "):
                        examine_targets[agent_name] = chosen_action.replace("examine ", "").strip()
                
                time.sleep(1)

            # --- BEAT 2: THE PHYSICS ENGINE ---
            obs, _, _, _ = self.env.step("step")
            
            # --- BEAT 3: VERIFIED PHYSICAL OUTCOMES ---
            self.parse_physical_outcomes(obs, self.tick_count)
            self.parse_and_print_locations(obs, self.tick_count)
            
            clean_narrative = self.filter_narrative(obs).strip()
            print("\n📜 NARRATIVE LOG:")
            print(clean_narrative if clean_narrative else "Nothing obvious happened.")
            
            # ---> CATCH EXAMINE DATA AND SAVE TO MEMORY <---
            for ex_agent, ex_target in examine_targets.items():
                if clean_narrative:
                    agent_obj = self.agents.get(ex_agent)
                    if agent_obj:
                        room = agent_obj.current_room
                        memory_string = f"{ex_agent} examines the {ex_target}. {clean_narrative}" # <--- Fixed!
                        
                        # Synchronous Database Write
                        self.db.log_broadcast(self.tick_count, room, "SYSTEM", [ex_agent], memory_string)
                        print(f"  🔍 [EXAMINE LOGGED] {ex_agent} saw: {clean_narrative}")
                        
                        # SAFE DATABASE CROSS-CHECK
                        try:
                            verify_mem = self.db.get_agent_context(ex_agent, limit=1)
                            print(f"  ✅ [DB VERIFY] Retrieved from memory: {verify_mem.strip()}")
                        except Exception as e:
                            print(f"  ⚠️ [DB VERIFY WARN] Data inserted, but verify fetch failed: {e}")
            
            # --- Parse and log the POST-ACTION sight data under the CURRENT tick ---
            sight_obs, _, _, _ = self.env.step("dump sight")
            current_sight_data = self.parse_sight_data(sight_obs, self.tick_count)
            
            # --- NEW: Fast-Forward Batching Logic ---
            if auto_ticks > 0:
                auto_ticks -= 1
                print(f"\n⏩ Auto-advancing... ({auto_ticks + 1} ticks remaining in this batch)")
                continue
            
            cmd = input("\nPress Enter for 1 tick, type a number to fast-forward, or 'q' to quit... ").strip()
            
            if cmd.lower() == 'q':
                print("Exiting simulation loop.")
                break
            elif cmd.isdigit():
                requested_ticks = int(cmd)
                if requested_ticks > 1:
                    # Subtract 1 because the loop immediately continues to the next tick
                    auto_ticks = requested_ticks - 1

def generate_dramabench_storyboard(session_id, db_logger, output_filename="Benchmark_Output.txt"):
    print(f"\n📝 Compiling Storyboard for Session: {session_id}...")
    
    all_npcs = db_logger.get_all_npcs()
    profile_lookup = {npc['name']: npc.get('persona', f"You are {npc['name']}.") for npc in all_npcs}
    
    response = db_logger.supabase.table('chat_logs')\
        .select('tick, room_name, sender, receiver, message')\
        .eq('session_id', session_id)\
        .order('tick')\
        .order('id').execute()
        
    records = response.data
    if not records:
        print("❌ No timeline data found for this session.")
        return

    # FIX: Group by (Tick, Room) to prevent fragmentation
    scenes_dict = {}
    
    for row in records:
        scene_key = (row['tick'], row['room_name'])
        
        if scene_key not in scenes_dict:
            scenes_dict[scene_key] = {
                "tick": row['tick'],
                "room": row['room_name'],
                "participants": set(),
                "dialogue_lines": []
            }
            
        scene_data = scenes_dict[scene_key]
        
        # FIX: Prevent Ghost Characters
        sender = row['sender']
        if sender not in ["ACTION", "SYSTEM"]:
            scene_data["participants"].add(sender)
            
            raw_receiver = row['receiver']
            if raw_receiver:
                # Handle stringified arrays from the database safely
                if isinstance(raw_receiver, str):
                    import re
                    # Extract just the alphabetic names
                    receivers = re.findall(r'[a-zA-Z\s]+', raw_receiver.replace('ALL', ''))
                    receivers = [r.strip() for r in receivers if r.strip()]
                else:
                    receivers = raw_receiver
                    
                for r in receivers:
                    # Validate against known profiles to eliminate single-letter garbage
                    if r != "ALL" and r in profile_lookup:
                        scene_data["participants"].add(r)
                
        scene_data["dialogue_lines"].append((sender, row['message']))

    # Write out the formatted Storyboard
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("=========================================\n")
        f.write("JERICHO STUDIO: OFFICIAL STORYBOARD\n")
        f.write(f"Session ID: {session_id}\n")
        f.write("=========================================\n")

        # Sort chronologically by tick, then alphabetically by room
        sorted_scenes = sorted(scenes_dict.values(), key=lambda x: (x['tick'], x['room']))

        for idx, scene in enumerate(sorted_scenes, 1):
            f.write(f"\nSCENE {idx}\n")
            f.write(f"LOCATION: {scene['room']}\n")
            f.write(f"TIME: Tick {scene['tick']}\n\n")
            
            f.write("SCENE DESCRIPTION:\n")
            f.write(f"The simulation takes place in the {scene['room']}.\n\n")
            
            f.write("CAST PROFILES:\n")
            if scene['participants']:
                for actor in sorted(scene['participants']):
                    desc = profile_lookup.get(actor, f"You are {actor}.")
                    f.write(f"- {actor.upper()}: {desc}\n")
            else:
                f.write("- NONE: Physical actions only.\n")
                
            f.write("\n-----------------------------------------\n")
            
            for sender, message in scene['dialogue_lines']:
                if sender == "ACTION" or sender == "SYSTEM":
                    f.write(f"*{message}*\n")
                else:
                    f.write(f"{sender}: {message}\n")

    print(f"✅ DramaBench storyboard compiled successfully: {output_filename}")

if __name__ == "__main__":
    import time
    
    GAME_FILE = "games/TeaHouseBetrayal.z8"  # Your compiled game file
    
    print("\n" + "="*50)
    print("🧪 BASELINE BENCHMARKING MANAGER")
    print("="*50)
    print("1. Start a FRESH benchmarking scenario (Isolated DB)")
    print("2. Resume an EXISTING benchmarking scenario")
    
    choice = input("\nSelect option (1/2): ").strip()
    
    if choice == '1':
        timestamp = int(time.time())
        SESSION_ID = f"BASELINE_BENCH_{timestamp}"
        print(f"\n[*] Booting fresh scenario. Data locked to: {SESSION_ID}")
    elif choice == '2':
        SESSION_ID = input("Enter EXACT Session ID to resume: ").strip()
        print(f"\n[*] Resuming timeline for: {SESSION_ID}")
    else:
        timestamp = int(time.time())
        SESSION_ID = f"BASELINE_BENCH_{timestamp}"
        print(f"[*] Defaulting to fresh scenario: {SESSION_ID}")

    try:
        runner = BaselineRunner(GAME_FILE, SESSION_ID)
        runner.run(max_ticks=50) 
    except KeyboardInterrupt:
        print("\nSimulation aborted by user.")
    except Exception as e:
        print(f"Simulation Error: {e}")
    finally:
        # Generate the Fountain script upon completion or manual exit
        generate_dramabench_storyboard(SESSION_ID, runner.db, f"{SESSION_ID}.txt")