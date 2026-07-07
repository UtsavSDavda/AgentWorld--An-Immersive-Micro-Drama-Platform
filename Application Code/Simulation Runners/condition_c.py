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
        
        # 3. New: Cognitive State Trackers
        self.last_action_result = None 
        self.current_objective = "Assess the situation and act according to my persona."
        self.immediate_blocker = "None."
        self.stress_level = "CALM"
        self.relational_stances = {}

    @property
    def full_persona(self):
        """Dynamically combines the core identity with current emotional states."""
        if not self.active_deltas:
            return self.base_persona
        # Append the accumulated deltas to the base persona
        return f"{self.base_persona} CURRENT MOOD: {' '.join(self.active_deltas)}"

    def reflect(self, session_id):
        """Evaluates recent memory to generate Persona Deltas and Cognitive States."""
        recent_events, recent_chats = self.get_split_memory(session_id, limit=5)
        
        if not recent_events and not recent_chats:
            return
        
        chat_memory = f"[RECENT PHYSICAL EVENTS]\n{recent_events}\n\n[RECENT CONVERSATION]\n{recent_chats}"

        prompt = f"""
        You are {self.name}.
        Your core persona is: {self.base_persona}
        Current Objective: {self.current_objective}
        
        RECENT EVENTS AND CONVERSATIONS:
        {chat_memory}
        
        TASK:
        Evaluate the recent events and update your internal state.
        1. Rate the 'Impact Level' of these recent events on your psyche from 1 (normal) to 10 (life-altering).
        2. If the impact is 6 or higher, write a 1-sentence 'Persona Delta' describing a mood shift. Else, leave blank.
        3. Update your 'current_objective' based on what just happened. Keep it actionable.
        4. Identify your 'immediate_blocker' (what is physically or socially stopping you right now).
        5. Set your 'stress_level' strictly to CALM, ALERT, or PANICKED.
        6. Update your 'relational_stances' mapping visible people to ALLY, NEUTRAL, or ADVERSARY.
        
        Respond ONLY with valid JSON in this exact format:
        {{
            "impact_score": 5,
            "persona_delta": "",
            "updated_states": {{
                "current_objective": "find a way to escape",
                "immediate_blocker": "the door is locked",
                "stress_level": "ALERT",
                "relational_stances": {{
                    "CharacterName": "ADVERSARY"
                }}
            }}
        }}
        """
        
        try:
            response = client.models.generate_content(
                model=MODEL_ID, 
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            
            data = json.loads(response.text)
            score = data.get("impact_score", 0)
            delta = data.get("persona_delta", "").strip()
            states = data.get("updated_states", {})
            
            # Update States
            self.current_objective = states.get("current_objective", self.current_objective)
            self.immediate_blocker = states.get("immediate_blocker", self.immediate_blocker)
            self.stress_level = states.get("stress_level", "CALM")
            self.relational_stances = states.get("relational_stances", self.relational_stances)
            
            print(f"  [🔄 STATE UPDATE] {self.name} | Stress: {self.stress_level} | Blocker: {self.immediate_blocker}")
            
            if score >= 6 and delta:
                print(f"  [💡 PSYCHOLOGICAL SHIFT] {self.name} experienced a high-impact event (Score: {score}/10)!")
                print(f"      Delta Added: '{delta}'")
                self.active_deltas.append(delta)
                if len(self.active_deltas) > 2:
                    self.active_deltas.pop(0)

        except Exception as e:
            print(f"  [!] Reflection failed for {self.name}: {e}")

    def generate_action_catalog(self, context_data):
        """Generates available Inform 7 actions."""
        catalog = ["wait"]
        if not context_data:
            return catalog

        for direction in context_data.get("direction", []):
            catalog.append(f"go {direction}")
            
        for person in context_data.get("person", []):
            catalog.append(f"examine {person}")
            catalog.append(f"talk to {person}") 
            
        for obj_data in context_data.get("object", []):
            name = obj_data["name"]
            state = obj_data["state"] 
            portability = obj_data["portability"]
            lock_status = obj_data.get("lock", "unlocked") 
            
            catalog.append(f"examine {name}")
            
            if portability == "portable":
                catalog.append(f"take {name}")
            if "device" in state:
                catalog.append(f"switch on {name}")
                catalog.append(f"switch off {name}")
            if "openable" in state:
                catalog.append(f"open {name}")
                catalog.append(f"close {name}")
                catalog.append(f"look inside {name}")
            if lock_status == "locked":
                catalog.append(f"unlock {name}")
                
        for inv_data in context_data.get("inventory", []):
            name = inv_data["name"]
            state = inv_data["state"]
            
            catalog.append(f"examine {name}")
            catalog.append(f"drop {name}")
            
            if "device" in state: 
                catalog.append(f"switch on {name}")
                catalog.append(f"switch off {name}")
            if "openable" in state:
                catalog.append(f"open {name}")
                catalog.append(f"close {name}")
                catalog.append(f"look inside {name}")

        return sorted(list(set(catalog)))
    
    def get_split_memory(self, session_id, limit=10):
        """Fetches memory and strictly splits physical events from dialogue."""
        try:
            response = self.db.supabase.table('chat_logs')\
                .select('tick, sender, receiver, message')\
                .eq('session_id', session_id)\
                .order('tick', desc=False).execute()
                
            events = []
            chats = []
            
            for row in response.data:
                sender = row['sender']
                raw_receiver = row['receiver']
                
                if isinstance(raw_receiver, str):
                    import re
                    receivers = [r.strip() for r in re.findall(r'[a-zA-Z\s]+', raw_receiver.replace('ALL', '')) if r.strip()]
                    if "ALL" in raw_receiver:
                        receivers.append("ALL")
                else:
                    receivers = raw_receiver if raw_receiver else []
                
                if sender == self.name or self.name in receivers or "ALL" in receivers:
                    if sender in ["ACTION", "SYSTEM"]:
                        events.append(f"[Tick {row['tick']}] {row['message']}")
                    else:
                        chats.append(f"[Tick {row['tick']}] {sender}: {row['message']}")
                        
            return "\n".join(events[-limit:]), "\n".join(chats[-limit:])
            
        except Exception as e:
            print(f"  [!] Memory fetch error for {self.name}: {e}")
            return "", ""

    def decide_action(self, context_data, session_id, debug_mode=True):
        """Prompts Gemini using Chain of Thought and Physics Troubleshooting."""
        catalog = self.generate_action_catalog(context_data)
        recent_events, recent_chats = self.get_split_memory(session_id, limit=10)
        
        visible_items = [obj["name"] for obj in context_data.get("object", [])]
        visible_people = context_data.get("person", [])
        my_inventory = [inv["name"] for inv in context_data.get("inventory", [])]
        
        stances_str = ", ".join([f"{k} ({v})" for k, v in self.relational_stances.items()]) if self.relational_stances else "None established."
        
        # Format the Feedback Warning & Troubleshooting Guide
        feedback_string = ""
        if self.last_action_result:
            if "fails" in self.last_action_result.lower() or "failed" in self.last_action_result.lower():
                feedback_string = f"\n[LAST ACTION RESULT]: {self.last_action_result}\n"
                feedback_string += "**TROUBLESHOOTING GUIDE:**\n"
                feedback_string += "Your last action failed. Do not repeat it. Instead, deduce the missing prerequisite:\n"
                feedback_string += "- If LOCKED: You need to find and hold the specific key, then use the 'unlock' command.\n"
                feedback_string += "- If HELD BY SOMEONE ELSE: You cannot 'take' it. You must use dialogue to convince them to 'drop' it.\n"
                feedback_string += "- If BLOCKED/UNKNOWN: Use the 'examine [object]' command to look for clues.\n"
            else:
                feedback_string = f"\n[LAST ACTION RESULT]: {self.last_action_result}\n"
        
        prompt = f"""
        You are {self.name}. 
        Persona: {self.full_persona}
        {feedback_string}
        
        CURRENT COGNITIVE STATE:
        Objective: {self.current_objective}
        Immediate Blocker: {self.immediate_blocker}
        Stress Level: {self.stress_level}
        Relational Stances: {stances_str}
        
        [RECENT PHYSICAL EVENTS] (What you just saw happen):
        {recent_events if recent_events else "No recent events."}
        
        [RECENT CONVERSATION] (What was just said in this room):
        {recent_chats if recent_chats else "No recent conversation."}
        
        CURRENT ENVIRONMENT:
        You are in: {self.current_room}
        People here: {', '.join(visible_people) if visible_people else 'No one'}
        Objects here: {', '.join(visible_items) if visible_items else 'Nothing'}
        Your Inventory: {', '.join(my_inventory) if my_inventory else 'You are holding nothing.'}
        
        CRITICAL RULES:
        1. You cannot physically 'take' an object that is held by another person. You must use dialogue to convince them to 'drop' it first.
        2. Choose EXACTLY ONE action from the AVAILABLE ACTIONS list.
        3. You must explain your logic first using a 'thought_process' before providing the exact 'action' string.
        
        AVAILABLE ACTIONS:
        {chr(10).join(catalog)}
        
        Respond ONLY with valid JSON in this exact format:
        {{
            "thought_process": "explain why you are choosing this action based on your state and blockers",
            "action": "exact string from catalog"
        }}
        """

        if debug_mode:
            print(f"\n[DEBUG: {self.name}'s Prompt] {'-'*40}")
            print(prompt.strip())
            print(f"[{'-'*60}]")

        try:
            response = client.models.generate_content(
                model=MODEL_ID, 
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            data = json.loads(response.text)
            chosen_action = data.get("action", "").strip()
            
            print(f"🧠 [DEBUG] {self.name} Thought: '{data.get('thought_process', '')}'")
            print(f"🧠 [DEBUG] {self.name} Selected: '{chosen_action}'\n")
            
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
        self.session_id = session_id
        
        db_manager = GameDBManager()
        db_name = db_manager.get_or_create_user_db(game_file, session_id)
        self.db = SQLLogger(session_id)
        
        self._initialize_agents_from_inform()
    
    def orchestrate_narrative(self, seed):
        """The Director LLM intercepts the simulation and forces cognitive updates."""
        print(f"\n🎬 [DIRECTOR ORCHESTRATOR] Processing Narrative Seed: '{seed}'")
        
        # 1. Gather current agent states
        agent_states = {}
        for name, agent in self.agents.items():
            agent_states[name] = {
                "persona": agent.base_persona,
                "current_objective": agent.current_objective,
                "stress_level": agent.stress_level,
                "relational_stances": agent.relational_stances
            }
        
        # 2. Build the Director Prompt
        prompt = f"""
        You are the Narrative Director for an interactive simulation.
        
        CURRENT AGENT STATES:
        {json.dumps(agent_states, indent=2)}
        
        NARRATIVE SEED (The plot pivot the user wants to enforce):
        "{seed}"
        
        TASK:
        Decide which agents need their internal cognitive states updated to execute this plot pivot.
        You must return a JSON dictionary containing ONLY the overrides for the affected agents.
        For each affected agent, you can update their 'current_objective', 'stress_level' (CALM, ALERT, or PANICKED), and 'relational_stances' (ALLY, NEUTRAL, or ADVERSARY).
        
        Respond ONLY with valid JSON in this exact format (only include agents whose states need to change):
        {{
            "AgentName": {{
                "current_objective": "New actionable objective based on the seed.",
                "stress_level": "PANICKED",
                "relational_stances": {{"OtherAgentName": "ADVERSARY"}}
            }}
        }}
        """
        
        try:
            # 3. Fire to Gemini
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            
            overrides = json.loads(response.text)
            
            # 4. Inject the parsed updates into the BaselineAgent objects
            print(f"  [📥 INJECTING COGNITIVE OVERRIDES]")
            for agent_name, new_state in overrides.items():
                if agent_name in self.agents:
                    agent = self.agents[agent_name]
                    if "current_objective" in new_state:
                        agent.current_objective = new_state["current_objective"]
                    if "stress_level" in new_state:
                        agent.stress_level = new_state["stress_level"]
                    if "relational_stances" in new_state:
                        agent.relational_stances = new_state["relational_stances"]
                    
                    print(f"    -> Override applied to {agent_name}:")
                    print(f"       Objective: '{agent.current_objective}'")
                    print(f"       Stress Level: {agent.stress_level}")
            
        except Exception as e:
            print(f"  [!] Director Orchestration failed: {e}")

    def get_occupants(self, room_name):
        """Returns a list of agent names currently standing in the specified room."""
        return [name for name, agent in self.agents.items() if agent.current_room == room_name]

    def _initialize_agents_from_inform(self):
        obs, _, _, _ = self.env.step("dump agents")
        pattern = re.compile(r"DATA_NPC:\s*(.*?)\s*\|\s*(.*?)(?=DATA_NPC|--- END NPC DATA ---|$)", re.DOTALL)
        
        print("\n--- Initializing AI Agents ---")
        for match in pattern.finditer(obs):
            name = match.group(1).strip()
            desc = match.group(2).strip()
            
            if not self.db.get_npc_profile(name):
                self.db.save_npc_profile(
                    name=name, raw_desc=desc, persona=f"You are {name}. {desc}", 
                    appearance=f"A portrait of {name}", gender="NEUTRAL", voice_id="en-US-Neural2-J"
                )
                
            self.agents[name] = BaselineAgent(name, self.db)
            print(f"Loaded {name} into system.")
    
    def naturalize_action(self, name, raw_action):
        action_text = raw_action[len(name):].strip() if raw_action.startswith(name) else raw_action.strip()
        parts = action_text.split(" ", 1)
        gerund = parts[0].lower()
        remainder = " " + parts[1] if len(parts) > 1 else ""

        gerund_map = {
            "going": "goes", "entering": "enters", "exiting": "exits", "getting": "gets",
            "jumping": "jumps", "climbing": "climbs", "taking": "takes", "dropping": "drops",
            "putting": "puts", "inserting": "inserts", "wearing": "wears", "removing": "removes",   
            "examining": "examines", "looking": "looks", "searching": "searches", "listening": "listens",
            "smelling": "smells", "tasting": "tastes", "touching": "touches", "opening": "opens",
            "closing": "closes", "locking": "locks", "unlocking": "unlocks", "switching": "switches",
            "pushing": "pushes", "pulling": "pulls", "turning": "turns", "tying": "ties",
            "setting": "sets", "asking": "asks", "telling": "tells", "answering": "answers",
            "saying": "says", "showing": "shows", "giving": "gives", "attacking": "attacks",
            "kissing": "kisses", "waking": "wakes", "waiting": "waits", "eating": "eats",
            "drinking": "drinks", "throwing": "throws", "rubbing": "rubs", "waving": "waves",
            "burning": "burns", "cutting": "cuts", "squeezing": "squeezes", "swinging": "swings",
            "sleeping": "sleeps"
        }

        if gerund in gerund_map:
            return f"{name} {gerund_map[gerund]}{remainder}."
        
        if gerund.endswith("ing"):
            base = gerund[:-3] 
            if base.endswith("y"):
                base = base[:-1] + "ies"
            elif base.endswith(("s", "sh", "ch", "x", "z")):
                base += "es"
            elif len(base) > 2 and base[-1] == base[-2]:
                 base = base[:-1] + "s"
            else:
                base += "s"
            return f"{name} {base}{remainder}."

        return f"{name} is {action_text}."

    def parse_sight_data(self, observation_text, tick_count):
        sight_dict = {}
        for line in observation_text.splitlines():
            line = line.strip()
            if line.startswith("DATA_SIGHT:"):
                self.db.log_engine_data(tick_count, "DATA_SIGHT", line)
                clean_line = line.replace("DATA_SIGHT:", "").strip()
                parts = [p.strip() for p in clean_line.split("|")]
                
                if len(parts) >= 3:
                    observer, category, target = parts[0], parts[1], parts[2]
                    
                    if observer not in sight_dict:
                        sight_dict[observer] = {"direction": [], "person": [], "object": [], "inventory": []}
                    
                    if category in ["object", "inventory"]:
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
            if clean_line.startswith(">"):
                continue
            if "executes plan:" in clean_line:
                continue
            if clean_line in ["Security Kiosk", "Server Corridor", "Data Vault", "Unknown", "Antechamber", "Inner Teahouse", "Courtyard"]:
                continue
            if any(clean_line.startswith(block) for block in skip_blocks):
                continue
                
            clean_lines.append(clean_line)
            
        return "\n".join(clean_lines)
    
    def parse_physical_outcomes(self, observation_text, tick_count):
        """Parses DATA_RESULT output to generate verified stage directions and feedback."""
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
                        stage_direction = self.naturalize_action(name, raw_action)
                        if name in self.agents:
                            self.agents[name].last_action_result = f"You successfully executed: {raw_action}"
                    else:
                        reason = parts[3] if len(parts) > 3 else "blocked"
                        clean_action = raw_action[len(name):].strip() if raw_action.startswith(name) else raw_action.strip()
                        stage_direction = f"{name} tries to {clean_action}, but fails ({reason})."
                        if name in self.agents:
                            self.agents[name].last_action_result = f"You tried to '{clean_action}', but failed ({reason})."
                    
                    room_occupants = self.get_occupants(room)
                    self.db.log_broadcast(tick_count, room, "ACTION", room_occupants, stage_direction)
                    
                    print(f"  🎬 [ACTION] {stage_direction}")

    def run(self, max_ticks=50):
        import time
        
        print("\n=== STARTING BASELINE SCENARIO ===")
        init_obs, _, _, _ = self.env.step("wait") 
        self.parse_and_print_locations(init_obs, self.tick_count) 
        obs, _, _, _ = self.env.step("dump sight")
        
        current_sight_data = self.parse_sight_data(obs, self.tick_count)
        
        auto_ticks = 0 
        
        while self.tick_count < max_ticks:
            self.tick_count += 1
            print(f"\n{'='*20} TICK {self.tick_count} {'='*20}")
            
            if self.tick_count % 3 == 0:
                print("\n🧘 Agents are reflecting on recent events...")
                for _, agent in self.agents.items():
                    agent.reflect(self.session_id)
            
            print("\n🧠 Agents are making decisions...")
            
            examine_targets = {}
            
            for agent_name, agent in self.agents.items():
                agent_context = current_sight_data.get(agent_name, {})
                chosen_action = agent.decide_action(agent_context, self.session_id, debug_mode=False)
                
                if chosen_action.startswith("talk to "):
                    target_person = chosen_action.replace("talk to ", "").strip()
                    chat_history = self.db.get_agent_context(agent_name)
                    
                    dialogue_line = generate_gemini_response(
                        sender=agent_name, receiver=[target_person], 
                        context=chat_history, room=agent.current_room, db_logger=self.db
                    )
                    
                    if dialogue_line:
                        room_occupants = self.get_occupants(agent.current_room)
                        self.db.log_broadcast(self.tick_count, agent.current_room, agent_name, room_occupants, dialogue_line)
                        
                        self.db.add_to_timeline(self.tick_count, agent.current_room)
                        print(f"  [DIALOGUE] {agent_name}: \"{dialogue_line}\"")
                    
                    self.env.step(f"force {agent_name} to wait")
                else:
                    self.env.step(f"force {agent_name} to {chosen_action}")
                    
                    if chosen_action.startswith("examine "):
                        examine_targets[agent_name] = chosen_action.replace("examine ", "").strip()
                
                time.sleep(1)

            obs, _, _, _ = self.env.step("step")
            
            self.parse_physical_outcomes(obs, self.tick_count)
            self.parse_and_print_locations(obs, self.tick_count)
            
            clean_narrative = self.filter_narrative(obs).strip()
            print("\n📜 NARRATIVE LOG:")
            print(clean_narrative if clean_narrative else "Nothing obvious happened.")
            
            for ex_agent, ex_target in examine_targets.items():
                if clean_narrative:
                    agent_obj = self.agents.get(ex_agent)
                    if agent_obj:
                        room = agent_obj.current_room
                        memory_string = f"{ex_agent} examines the {ex_target}. {clean_narrative}" 
                        
                        self.db.log_broadcast(self.tick_count, room, "SYSTEM", [ex_agent], memory_string)
                        print(f"  🔍 [EXAMINE LOGGED] {ex_agent} saw: {clean_narrative}")
                        
                        try:
                            verify_mem = self.db.get_agent_context(ex_agent, limit=1)
                        except Exception as e:
                            print(f"  ⚠️ [DB VERIFY WARN] Data inserted, but verify fetch failed: {e}")
            
            sight_obs, _, _, _ = self.env.step("dump sight")
            current_sight_data = self.parse_sight_data(sight_obs, self.tick_count)
            
            if auto_ticks > 0:
                auto_ticks -= 1
                print(f"\n⏩ Auto-advancing... ({auto_ticks + 1} ticks remaining in this batch)")
                continue
            
            # Step 2: Modified input hook for the Orchestrator
            while True:
                cmd = input("\nPress Enter for 1 tick, a number to fast-forward, 'seed: <plot>' to orchestrate, or 'q' to quit... ").strip()
                
                if cmd.lower().startswith("seed:"):
                    seed_text = cmd[5:].strip()
                    self.orchestrate_narrative(seed_text)
                    continue  # Traps the user back at the input prompt without incrementing ticks
                
                if cmd.lower() == 'q':
                    print("Exiting simulation loop.")
                    return
                elif cmd.isdigit():
                    requested_ticks = int(cmd)
                    if requested_ticks > 1:
                        auto_ticks = requested_ticks - 1
                break  # Standard flow to break the input loop and proceed to the next tick

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
        
        sender = row['sender']
        if sender not in ["ACTION", "SYSTEM"]:
            scene_data["participants"].add(sender)
            
            raw_receiver = row['receiver']
            if raw_receiver:
                if isinstance(raw_receiver, str):
                    import re
                    receivers = re.findall(r'[a-zA-Z\s]+', raw_receiver.replace('ALL', ''))
                    receivers = [r.strip() for r in receivers if r.strip()]
                else:
                    receivers = raw_receiver
                    
                for r in receivers:
                    if r != "ALL" and r in profile_lookup:
                        scene_data["participants"].add(r)
                
        scene_data["dialogue_lines"].append((sender, row['message']))

    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("=========================================\n")
        f.write("JERICHO STUDIO: OFFICIAL STORYBOARD\n")
        f.write(f"Session ID: {session_id}\n")
        f.write("=========================================\n")

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
    
    GAME_FILE = "games/TheCipherBlackSite1.z8"  
    
    print("\n" + "="*50)
    print("🧪 UPGRADED BENCHMARKING MANAGER (CONDITION C)")
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
        generate_dramabench_storyboard(SESSION_ID, runner.db, f"{SESSION_ID}.txt")