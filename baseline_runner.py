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
        chat_memory = self.db.get_agent_context(self.name, limit=5)
        
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
        """Builds a verified list of verbs, specifically adding social actions."""
        catalog = ["wait"]
        if not context_data:
            return catalog

        # Movement
        for direction in context_data.get("direction", []):
            catalog.append(f"go {direction}")
            
        # Social Actions - This gives the agent the CHOICE to talk
        for person in context_data.get("person", []):
            catalog.append(f"examine {person}")
            catalog.append(f"talk to {person}") 
            
        # Object Actions
        for obj, state in context_data.get("object", []):
            catalog.append(f"examine {obj}")
            catalog.append(f"take {obj}")
            if state == "device_off": catalog.append(f"switch on {obj}")
            elif state == "device_on": catalog.append(f"switch off {obj}")
            elif state == "openable_closed": catalog.append(f"open {obj}")
            elif state == "openable_open": 
                catalog.append(f"close {obj}")
                catalog.append(f"look inside {obj}")
                
        # Inventory Actions
        for item, state in context_data.get("inventory", []):
            catalog.append(f"examine {item}")
            catalog.append(f"drop {item}")

        return sorted(list(set(catalog)))

    def decide_action(self, context_data, debug_mode=True):
        """Prompts Gemini using the dynamic full_persona and Chat Memory."""
        catalog = self.generate_action_catalog(context_data)
        
        # Fetch Chat Memory from Database
        chat_memory = self.db.get_agent_context(self.name, limit=5)
        
        visible_items = [obj[0] for obj in context_data.get("object", [])]
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

        # --- DEBUG LOGGING: SEE THE LLM'S BRAIN ---
        if debug_mode:
            print(f"\n[DEBUG: {self.name}'s Prompt] {'-'*40}")
            print(prompt.strip())
            print(f"[{'-'*60}]")

        try:
            response = client.models.generate_content(model=MODEL_ID, contents=prompt)
            chosen_action = response.text.strip()
            
            if debug_mode:
                print(f"[DEBUG: {self.name}'s Raw Output] -> '{chosen_action}'")
            
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

    def parse_sight_data(self, observation_text):
        sight_dict = {}
        for line in observation_text.splitlines():
            line = line.strip()
            if line.startswith("DATA_SIGHT:"):
                clean_line = line.replace("DATA_SIGHT:", "").strip()
                parts = [p.strip() for p in clean_line.split("|")]
                if len(parts) >= 3:
                    observer, category, target = parts[0], parts[1], parts[2]
                    state = parts[3] if len(parts) >= 4 else "basic"
                    
                    if observer not in sight_dict:
                        sight_dict[observer] = {"direction": [], "person": [], "object": [], "inventory": []}
                    
                    if category in ["object", "inventory"]:
                        sight_dict[observer][category].append((target, state))
                    else:
                        sight_dict[observer][category].append(target)
        return sight_dict

    def parse_and_print_locations(self, observation_text):
        locations = {}
        print("\n📍 CURRENT LOCATIONS:")
        for line in observation_text.splitlines():
            line = line.strip()
            if line.startswith("DATA_LOC:"):
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
        skip_blocks = ["--- START DATA ---", "--- END DATA ---", "--- START SIGHT ---", "--- END SIGHT ---", "--- Simulation Step ---", "DATA_RESULT", "DATA_EVENT"]
        for line in observation_text.splitlines():
            if not any(line.startswith(block) for block in skip_blocks) and line.strip():
                clean_lines.append(line)
        return "\n".join(clean_lines)

    def run(self, max_ticks=20):
        import time # Ensure time is imported for rate limiting
        
        print("\n=== STARTING BASELINE SIMULATION ===")
        obs, _, _, _ = self.env.step("dump sight")
        
        while self.tick_count < max_ticks:
            self.tick_count += 1
            print(f"\n{'='*20} TICK {self.tick_count} {'='*20}")
            
            current_sight_data = self.parse_sight_data(obs)
            
            # --- THE NEW REFLECTION LOOP ---
            # Trigger psychological reflection every 3 ticks
            if self.tick_count % 3 == 0:
                print("\n🧘 Agents are reflecting on recent events...")
                for agent_name, agent in self.agents.items():
                    agent.reflect()
            
            print("\n🧠 Agents are making decisions...")
            for agent_name, agent in self.agents.items():
                agent_context = current_sight_data.get(agent_name, {})
                
                # We pass debug_mode=True to expose the LLM's inner monologue to your terminal
                chosen_action = agent.decide_action(agent_context, debug_mode=True)
                
                # --- INTERCEPT CHAT ACTIONS ---
                if chosen_action.startswith("talk to "):
                    target_person = chosen_action.replace("talk to ", "").strip()
                    print(f"  > 🗣️ {agent_name} chooses to chat with {target_person}!")
                    
                    # 1. Fetch memory to continue conversation
                    chat_history = self.db.get_agent_context(agent_name)
                    
                    # 2. Generate the dialogue line using your existing function
                    dialogue_line = generate_gemini_response(
                        sender=agent_name, 
                        receiver=[target_person], 
                        context=chat_history, 
                        room=agent.current_room, 
                        db_logger=self.db
                    )
                    
                    # 3. Log to database so it appears in memory next turn
                    if dialogue_line:
                        self.db.log_broadcast(self.tick_count, agent.current_room, agent_name, [target_person], dialogue_line)
                        print(f"      [{agent_name}]: \"{dialogue_line}\"")
                    
                    # 4. Force Inform 7 to 'wait' so physics engine doesn't break
                    self.env.step(f"force {agent_name} to wait")
                    
                # --- HANDLE NORMAL GAME ACTIONS ---
                else:
                    print(f"  > 🏃 {agent_name} decides to: {chosen_action}")
                    self.env.step(f"force {agent_name} to {chosen_action}")
                
                time.sleep(1) # Prevent hitting Gemini API rate limits

            # --- EXECUTE THE SIMULATION TICK ---
            obs, _, _, _ = self.env.step("step")
            
            # Print physical tracking
            self.parse_and_print_locations(obs)
            
            # Print the clean Inform 7 narrative
            print("\n📜 NARRATIVE LOG:")
            print(self.filter_narrative(obs))
            
            # Prepare sight data for the next loop
            obs, _, _, _ = self.env.step("dump sight")
            
            # Manual pause so you can read the debug prompts
            cmd = input("\nPress Enter to proceed to the next tick (or type 'q' to quit)... ")
            if cmd.lower() == 'q':
                print("Exiting simulation loop.")
                break

if __name__ == "__main__":
    GAME_FILE = "games/Will5.z8"  # Your compiled game file
    SESSION_ID = "baseline_test_02" # Required for DB isolation
    
    try:
        runner = BaselineRunner(GAME_FILE, SESSION_ID)
        runner.run(max_ticks=50)
    except KeyboardInterrupt:
        print("\nSimulation aborted by user.")
    except Exception as e:
        print(f"Simulation Error: {e}")