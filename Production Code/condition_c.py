import jericho
import re
import time
import os
from google import genai
from dotenv import load_dotenv
import json
from google.genai import types
import pickle
import base64
from chat_logger import GameDBManager, SQLLogger
from videoprompts import SYSTEM_PROMPT

load_dotenv()
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
        
        # 3. Cognitive State Trackers
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
        return f"{self.base_persona} CURRENT MOOD: {' '.join(self.active_deltas)}"

    def reflect(self, session_id, client):
        """Evaluates recent memory using a dynamic client to update state machines."""
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
                self.active_deltas.append(delta)
                if len(self.active_deltas) > 2:
                    self.active_deltas.pop(0)

        except Exception as e:
            print(f"  [!] Reflection failed for {self.name}: {e}")

    def generate_action_catalog(self, context_data):
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

    def decide_action(self, context_data, session_id, client, debug_mode=True):
        """Prompts Gemini using Chain of Thought powered by a dynamic worker client instance."""
        catalog = self.generate_action_catalog(context_data)
        recent_events, recent_chats = self.get_split_memory(session_id, limit=10)
        
        visible_items = [obj["name"] for obj in context_data.get("object", [])]
        visible_people = context_data.get("person", [])
        my_inventory = [inv["name"] for inv in context_data.get("inventory", [])]
        
        stances_str = ", ".join([f"{k} ({v})" for k, v in self.relational_stances.items()]) if self.relational_stances else "None established."
        
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
        
        [RECENT PHYSICAL EVENTS]:
        {recent_events if recent_events else "No recent events."}
        
        [RECENT CONVERSATION]:
        {recent_chats if recent_chats else "No recent conversation."}
        
        CURRENT ENVIRONMENT:
        You are in: {self.current_room}
        People here: {', '.join(visible_people) if visible_people else 'No one'}
        Objects here: {', '.join(visible_items) if visible_items else 'Nothing'}
        Your Inventory: {', '.join(my_inventory) if my_inventory else 'You are holding nothing.'}
        
        AVAILABLE ACTIONS:
        {chr(10).join(catalog)}
        
        Respond ONLY with valid JSON in this exact format:
        {{
            "thought_process": "explain why you are choosing this action based on your state and blockers",
            "action": "exact string from catalog"
        }}
        """

        try:
            response = client.models.generate_content(
                model=MODEL_ID, 
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            data = json.loads(response.text)
            chosen_action = data.get("action", "").strip()
            
            if chosen_action not in catalog:
                return "wait"
            return chosen_action
            
        except Exception as e:
            print(f"  [!] API Error for {self.name}: {e}. Defaulting to 'wait'.")
            return "wait"


class BaselineRunner:
    def __init__(self, game_file, session_id):
        print(f"Loading game environment: {game_file}...")
        self.env = jericho.FrotzEnv(game_file)
        self.session_id = session_id
        self.game_name = os.path.basename(game_file)
        
        # --- ADD THESE TWO LINES ---
        self.tick_count = 0
        self.agents = {} 
        # ---------------------------
        
        db_manager = GameDBManager()
        db_name = db_manager.get_or_create_user_db(game_file, session_id)
        self.db = SQLLogger(session_id)
        self.logger = self.db # Alias bridge for rendering pipeline compatibility
        
        self._initialize_agents_from_inform()
        
        # Cloud Isolation Recovery Implementation
        saved_state_b64 = self.db.get_zmachine_state()
        if saved_state_b64:
            self.load_game(saved_state_b64)
        else:
            self.tick_count = 0
            self.env.reset()
            print(f"🆕 Started fresh cloud simulation for session {session_id}")
            
        # Ground and sync environmental truth matrices
        init_obs, _, _, _ = self.env.step("wait") 
        self.parse_and_print_locations(init_obs, self.tick_count) 
        obs, _, _, _ = self.env.step("dump sight")
        self.current_sight_data = self.parse_sight_data(obs, self.tick_count)

    def save_game(self):
        """Serializes and flushes raw engine runtime matrices directly to Supabase."""
        state_tuple = self.env.get_state()
        pickled_bytes = pickle.dumps(state_tuple)
        b64_state = base64.b64encode(pickled_bytes).decode('utf-8')
        
        self.db.save_zmachine_state(b64_state)
        self.db.save_current_tick(self.tick_count)
        print(f"💾 Cloud state synchronized to cluster at Tick {self.tick_count}")

    def load_game(self, b64_state):
        """Restores process memory vectors from network boundaries inside distributed architectures."""
        try:
            pickled_bytes = base64.b64decode(b64_state)
            state_tuple = pickle.loads(pickled_bytes)
            
            self.env.set_state(state_tuple)
            self.tick_count = self.db.get_saved_tick()
            print(f"📂 Execution frames restored. Resuming thread checkpoint at Tick {self.tick_count}")
        except Exception as e:
            print(f"⚠️ State synchronization anomaly. Resetting container memory vectors. Error: {e}")
            self.tick_count = 0
            self.env.reset()

    def step(self, api_key=None):
        """Ticks the simulation forward exactly 1 turn using bounded context isolation handles."""
        self.tick_count += 1
        
        # Instantiate localized worker client
        effective_key = api_key if api_key else os.getenv("GEMINI_API_KEY")
        local_client = genai.Client(api_key=effective_key)
        
        if self.tick_count % 3 == 0:
            print("\n🧘 Agents are reflecting on recent events...")
            for _, agent in self.agents.items():
                agent.reflect(self.session_id, local_client)
        
        print("\n🧠 Agents are making decisions...")
        examine_targets = {}
        
        for agent_name, agent in self.agents.items():
            agent_context = self.current_sight_data.get(agent_name, {})
            chosen_action = agent.decide_action(agent_context, self.session_id, local_client, debug_mode=False)
            
            if chosen_action.startswith("talk to "):
                target_person = chosen_action.replace("talk to ", "").strip()
                chat_history = self.db.get_agent_context(agent_name)
                
                dialogue_line = self._generate_dynamic_gemini_response(
                    local_client, agent_name, [target_person], chat_history, agent.current_room
                )
                
                if dialogue_line:
                    room_occupants = self.get_occupants(agent.current_room)
                    self.db.log_broadcast(self.tick_count, agent.current_room, agent_name, room_occupants, dialogue_line)
                    self.db.add_to_timeline(self.tick_count, agent.current_room)
                
                self.env.step(f"force {agent_name} to wait")
            else:
                self.env.step(f"force {agent_name} to {chosen_action}")
                if chosen_action.startswith("examine "):
                    examine_targets[agent_name] = chosen_action.replace("examine ", "").strip()
            
            time.sleep(0.5)

        obs, _, _, _ = self.env.step("step")
        self.parse_physical_outcomes(obs, self.tick_count)
        locations = self.parse_and_print_locations(obs, self.tick_count)
        
        clean_narrative = self.filter_narrative(obs).strip()
        for ex_agent, ex_target in examine_targets.items():
            if clean_narrative:
                agent_obj = self.agents.get(ex_agent)
                if agent_obj:
                    memory_string = f"{ex_agent} examines the {ex_target}. {clean_narrative}" 
                    self.db.log_broadcast(self.tick_count, agent_obj.current_room, "SYSTEM", [ex_agent], memory_string)
        
        sight_obs, _, _, _ = self.env.step("dump sight")
        self.current_sight_data = self.parse_sight_data(sight_obs, self.tick_count)
        
        # Enforce cloud persistence checkpointing
        self.save_game()
        
        # Assemble localized monitor telemetry payloads
        agent_telemetry = {}
        for name, agent in self.agents.items():
            agent_telemetry[name] = {
                "room": agent.current_room,
                "objective": agent.current_objective,
                "blocker": agent.immediate_blocker,
                "stress": agent.stress_level
            }
            
        return {"locations": locations, "states": agent_telemetry}

    def _generate_dynamic_gemini_response(self, client, sender, receiver, context, room):
        profile = self.db.get_npc_profile(sender)
        personality = profile[0] if profile else f"You are {sender}."
        
        formatted_prompt = SYSTEM_PROMPT.format(
            sender=sender,
            personality=personality,
            room=room,
            receivers=receiver,
            context=context if context else "(No previous conversation)"
        )

        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=MODEL_ID, contents=formatted_prompt
                )
                return response.text.strip().replace(f"{sender}:", "").replace('"', '')
            except Exception as e:
                time.sleep(2 ** attempt)
        return None

    def orchestrate_narrative(self, seed, api_key=None):
        effective_key = api_key if api_key else os.getenv("GEMINI_API_KEY")
        local_client = genai.Client(api_key=effective_key)
        
        agent_states = {}
        for name, agent in self.agents.items():
            agent_states[name] = {
                "persona": agent.base_persona,
                "current_objective": agent.current_objective,
                "stress_level": agent.stress_level,
                "relational_stances": agent.relational_stances
            }
        
        prompt = f"""
        You are the Narrative Director for an interactive simulation.
        CURRENT AGENT STATES:
        {json.dumps(agent_states, indent=2)}
        
        NARRATIVE SEED: "{seed}"
        
        TASK:
        Return a JSON overrides layout detailing structural frame changes.
        Format layout output strictly:
        {{
            "AgentName": {{
                "current_objective": "...",
                "stress_level": "PANICKED/ALERT/CALM",
                "relational_stances": {{"Target": "ADVERSARY"}}
            }}
        }}
        """
        try:
            response = local_client.models.generate_content(
                model=MODEL_ID, contents=prompt, config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            overrides = json.loads(response.text)
            for agent_name, new_state in overrides.items():
                if agent_name in self.agents:
                    agent = self.agents[agent_name]
                    agent.current_objective = new_state.get("current_objective", agent.current_objective)
                    agent.stress_level = new_state.get("stress_level", agent.stress_level)
                    agent.relational_stances = new_state.get("relational_stances", agent.relational_stances)
            self.save_game()
        except Exception as e:
            print(f"Director runtime intervention trace abort: {e}")

    def get_occupants(self, room_name):
        return [name for name, agent in self.agents.items() if agent.current_room == room_name]

    def _initialize_agents_from_inform(self):
        obs, _, _, _ = self.env.step("dump agents")
        pattern = re.compile(r"DATA_NPC:\s*(.*?)\s*\|\s*(.*?)(?=DATA_NPC|--- END NPC DATA ---|$)", re.DOTALL)
        for match in pattern.finditer(obs):
            name = match.group(1).strip()
            desc = match.group(2).strip()
            if not self.db.get_npc_profile(name):
                self.db.save_npc_profile(name=name, raw_desc=desc, persona=f"You are {name}. {desc}", appearance=f"A portrait of {name}", gender="NEUTRAL", voice_id="en-US-Neural2-J")
            self.agents[name] = BaselineAgent(name, self.db)
    
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
            "burning": "burns", "cutting": "cuts", "squeezing": "squeezes", "swinging": "swings", "sleeping": "sleeps"
        }
        if gerund in gerund_map: return f"{name} {gerund_map[gerund]}{remainder}."
        if gerund.endswith("ing"):
            base = gerund[:-3] 
            if base.endswith("y"): base = base[:-1] + "ies"
            elif base.endswith(("s", "sh", "ch", "x", "z")): base += "es"
            elif len(base) > 2 and base[-1] == base[-2]: base = base[:-1] + "s"
            else: base += "s"
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
                    if observer not in sight_dict: sight_dict[observer] = {"direction": [], "person": [], "object": [], "inventory": []}
                    if category in ["object", "inventory"]:
                        sight_dict[observer][category].append({
                            "name": target, "state": parts[3] if len(parts) >= 4 else "basic",
                            "portability": parts[4] if len(parts) >= 5 else "portable", "lock": parts[5] if len(parts) >= 6 else "unlocked"
                        })
                    else:
                        sight_dict[observer][category].append(target)
        return sight_dict

    def parse_and_print_locations(self, observation_text, tick_count):
        locations = {}
        for line in observation_text.splitlines():
            line = line.strip()
            if line.startswith("DATA_LOC:"):
                self.db.log_engine_data(tick_count, "DATA_LOC", line)
                parts = [p.strip() for p in line.replace("DATA_LOC:", "").strip().split("|")]
                if len(parts) == 2:
                    locations[parts[0]] = parts[1]
                    if parts[0] in self.agents: self.agents[parts[0]].current_room = parts[1]
        return locations

    def filter_narrative(self, observation_text):
        clean_lines = []
        skip_blocks = ["--- START DATA ---", "--- END DATA ---", "--- START SIGHT ---", "--- END SIGHT ---", "--- START ROOM DATA ---", "--- END ROOM DATA ---", "--- BEGIN NPC DATA ---", "--- END NPC DATA ---", "--- Simulation Step ---", "DATA_RESULT", "DATA_EVENT", "DATA_LOC", "DATA_SIGHT", "DATA_NPC", "DATA_ROOM"]
        for line in observation_text.splitlines():
            clean_line = line.strip()
            if not clean_line or clean_line.startswith(">") or "executes plan:" in clean_line: continue
            if clean_line in ["Security Kiosk", "Server Corridor", "Data Vault", "Unknown", "Antechamber", "Inner Teahouse", "Courtyard"]: continue
            if any(clean_line.startswith(block) for block in skip_blocks): continue
            clean_lines.append(clean_line)
        return "\n".join(clean_lines)
    
    def parse_physical_outcomes(self, observation_text, tick_count):
        for line in observation_text.splitlines():
            line = line.strip()
            if line.startswith("DATA_RESULT:"):
                self.db.log_engine_data(tick_count, "DATA_RESULT", line)
                parts = [p.strip() for p in line.replace("DATA_RESULT:", "").strip().split("|")]
                if len(parts) >= 3:
                    name, status, raw_action = parts[0], parts[1], parts[2]
                    if status == "SUCCESS":
                        stage_direction = self.naturalize_action(name, raw_action)
                        if name in self.agents: self.agents[name].last_action_result = f"You successfully executed: {raw_action}"
                    else:
                        reason = parts[3] if len(parts) > 3 else "blocked"
                        clean_action = raw_action[len(name):].strip() if raw_action.startswith(name) else raw_action.strip()
                        stage_direction = f"{name} tries to {clean_action}, but fails ({reason})."
                        if name in self.agents: self.agents[name].last_action_result = f"You tried to '{clean_action}', but failed ({reason})."
                    self.db.log_broadcast(tick_count, self.agents[name].current_room if name in self.agents else "Unknown", "ACTION", self.get_occupants(self.agents[name].current_room if name in self.agents else "Unknown"), stage_direction)