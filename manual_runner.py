import jericho
import re
import time
import os
import sqlite3
from dotenv import load_dotenv

# Import your existing database logger and chat generation functions
from chat_logger import SQLLogger, GameDBManager

load_dotenv()

class ManualRunner:
    def __init__(self, game_file, session_id):
        print(f"Loading game environment: {game_file}...")
        self.env = jericho.FrotzEnv(game_file)
        self.tick_count = 0
        self.npc_names = []
        self.agent_rooms = {} 
        
        # Initialize Database Logger
        db_manager = GameDBManager()
        db_name = db_manager.get_or_create_user_db(game_file, session_id)
        self.db = SQLLogger(session_id)
        
    def parse_locations(self, observation_text):
        locations = {}
        for line in observation_text.splitlines():
            line = line.strip()
            if line.startswith("DATA_LOC:"):
                self.db.log_engine_data(self.tick_count, "DATA_LOC", line)
                clean_line = line.replace("DATA_LOC:", "").strip()
                parts = [p.strip() for p in clean_line.split("|")]
                if len(parts) == 2:
                    name, room = parts[0], parts[1]
                    locations[name] = room
                    self.agent_rooms[name] = room
        return locations

    def parse_room_data(self, observation_text):
        pattern = re.compile(
            r"DATA_ROOM:\s*(.*?)\s*\|\s*(.*?)(?=DATA_ROOM|--- END ROOM DATA ---|$)",
            re.DOTALL
        )
        for match in pattern.finditer(observation_text):
            r_name = match.group(1).strip()
            r_desc = match.group(2).strip()
            self.db.update_room_desc(r_name, r_desc, self.tick_count)

    def parse_npc_data(self, observation_text):
        pattern = re.compile(
            r"DATA_NPC:\s*(.*?)\s*\|\s*(.*?)(?=DATA_NPC|--- END NPC DATA ---|$)",
            re.DOTALL
        )
        for match in pattern.finditer(observation_text):
            npc_name = match.group(1).strip()
            raw_desc = match.group(2).strip()
            
            if npc_name not in self.npc_names:
                self.npc_names.append(npc_name)
                
            if not self.db.get_npc_profile(npc_name):
                self.db.save_npc_profile(
                    name=npc_name, raw_desc=raw_desc, persona=f"You are {npc_name}. {raw_desc}", 
                    appearance=f"A portrait of {npc_name}", gender="NEUTRAL", voice_id="en-US-Neural2-J"
                )

    def update_world_state(self):
        obs_rooms, _, _, _ = self.env.step("dump rooms")
        self.parse_room_data(obs_rooms)
        
        obs_npcs, _, _, _ = self.env.step("dump agents")
        self.parse_npc_data(obs_npcs)

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

    def naturalize_action(self, name, raw_action):
        action_text = raw_action[len(name):].strip() if raw_action.startswith(name) else raw_action.strip()
        parts = action_text.split(" ", 1)
        gerund = parts[0].lower()
        remainder = " " + parts[1] if len(parts) > 1 else ""

        gerund_map = {
            "going": "goes", "entering": "enters", "exiting": "exits", "getting": "gets",
            "taking": "takes", "dropping": "drops", "putting": "puts", "inserting": "inserts",
            "examining": "examines", "looking": "looks", "searching": "searches",
            "opening": "opens", "closing": "closes", "locking": "locks", "unlocking": "unlocks",
            "switching": "switches", "pushing": "pushes", "pulling": "pulls", "turning": "turns",
            "asking": "asks", "telling": "tells", "answering": "answers", "saying": "says",
            "waiting": "waits", "eating": "eats", "drinking": "drinks"
        }

        if gerund in gerund_map:
            return f"{name} {gerund_map[gerund]}{remainder}."
        
        if gerund.endswith("ing"):
            base = gerund[:-3]
            if base.endswith("y"): base = base[:-1] + "ies"
            elif base.endswith(("s", "sh", "ch", "x", "z")): base += "es"
            elif len(base) > 2 and base[-1] == base[-2]: base = base[:-1] + "s"
            else: base += "s"
            return f"{name} {base}{remainder}."

        return f"{name} is {action_text}."

    def parse_physical_outcomes(self, observation_text, tick_count):
        """Parses DATA_RESULT output to generate verified stage directions and logs them."""
        for line in observation_text.splitlines():
            line = line.strip()
            if line.startswith("DATA_RESULT:"):
                # Save ground truth to engine_logs
                self.db.log_engine_data(tick_count, "DATA_RESULT", line)
                
                clean = line.replace("DATA_RESULT:", "").strip()
                parts = [p.strip() for p in clean.split("|")]
                
                if len(parts) >= 3:
                    name = parts[0]
                    status = parts[1]
                    raw_action = parts[2] 
                    
                    room = self.agent_rooms.get(name, "Unknown")
                    
                    if status == "SUCCESS":
                        stage_direction = self.naturalize_action(name, raw_action)
                        color = "\033[92m" # Green
                    else:
                        reason = parts[3] if len(parts) > 3 else "blocked"
                        clean_action = raw_action[len(name):].strip() if raw_action.startswith(name) else raw_action.strip()
                        stage_direction = f"{name} tries to {clean_action}, but fails ({reason})."
                        color = "\033[91m" # Red
                    
                    # Log natural narrative to chat_logs
                    self.db.log_broadcast(tick_count, room, "ACTION", ["ALL"], stage_direction)
                    
                    # Print clearly to the manual tester terminal
                    reset = "\033[0m"
                    print(f"\n  🎬 [{color}{status}{reset}] {stage_direction}")

    def filter_narrative(self, observation_text):
        clean_lines = []
        skip_blocks = ["--- START DATA ---", "--- END DATA ---", "--- START SIGHT ---", "--- END SIGHT ---", "--- Simulation Step ---", "DATA_RESULT", "DATA_EVENT", "DATA_LOC"]
        for line in observation_text.splitlines():
            if not any(line.startswith(block) for block in skip_blocks) and line.strip():
                clean_lines.append(line)
        return "\n".join(clean_lines)

    def run(self):
        print("\n=== STARTING MANUAL TESTING SCENARIO ===")
        self.update_world_state()
        
        init_obs, _, _, _ = self.env.step("wait") 
        self.parse_locations(init_obs)
        
        while True:
            self.tick_count += 1
            print(f"\n{'='*20} TICK {self.tick_count} {'='*20}")
            
            # Get current sight data
            obs_sight, _, _, _ = self.env.step("dump sight")
            current_sight_data = self.parse_sight_data(obs_sight, self.tick_count)
            
            # 1. Select Agent
            print("\nSelect an agent to control:")
            for i, name in enumerate(self.npc_names):
                print(f"{i+1}. {name} (in {self.agent_rooms.get(name, 'Unknown')})")
            print("0. Step Simulation (All agents wait)")
            print("q. Quit")
            
            agent_choice = input("\nChoice: ").strip()
            
            if agent_choice.lower() == 'q':
                break
                
            if agent_choice == '0':
                obs, _, _, _ = self.env.step("step")
                self.parse_physical_outcomes(obs, self.tick_count)
                self.parse_locations(obs)
                print("\n📜 NARRATIVE LOG:")
                print(self.filter_narrative(obs))
                continue
                
            try:
                agent_idx = int(agent_choice) - 1
                if 0 <= agent_idx < len(self.npc_names):
                    active_agent = self.npc_names[agent_idx]
                else:
                    print("Invalid choice.")
                    self.tick_count -= 1
                    continue
            except ValueError:
                print("Invalid input.")
                self.tick_count -= 1
                continue

            # 2. Show Action Menu for Selected Agent
            agent_context = current_sight_data.get(active_agent, {})
            catalog = self.generate_action_catalog(agent_context)
            
            print(f"\n--- Actions for {active_agent} ---")
            for i, action in enumerate(catalog):
                print(f"{i+1}. {action}")
                
            cmd = input(f"\nSelect action number (1-{len(catalog)}) or type custom command: ").strip()
            
            if cmd.isdigit():
                cmd_idx = int(cmd) - 1
                if 0 <= cmd_idx < len(catalog):
                    action = catalog[cmd_idx]
                else:
                    print("Invalid action number. Defaulting to wait.")
                    action = "wait"
            else:
                action = cmd
                
            # 3. Execute Action
            if action.startswith("talk to "):
                target_person = action.replace("talk to ", "").strip()
                message = input(f"Enter dialogue for {active_agent} to say to {target_person}: ")
                
                room = self.agent_rooms.get(active_agent, "Unknown")
                self.db.log_broadcast(self.tick_count, room, active_agent, [target_person], message)
                self.db.add_to_timeline(self.tick_count, room)
                
                print(f"  [DIALOGUE LOGGED] {active_agent}: \"{message}\"")
                self.env.step(f"force {active_agent} to wait")
            else:
                self.env.step(f"force {active_agent} to {action}")
                
            # 4. Step Simulation and Get Results
            obs, _, _, _ = self.env.step("step")
            self.parse_physical_outcomes(obs, self.tick_count)
            self.parse_locations(obs)
            
            print("\n📜 NARRATIVE LOG:")
            print(self.filter_narrative(obs))

if __name__ == "__main__":
    GAME_FILE = "games/TeaHouseBetrayal.z8"  # Ensure this points to your compiled game
    
    print("\n" + "="*50)
    print("🧪 MANUAL BENCHMARKING TOOL")
    print("="*50)
    
    timestamp = int(time.time())
    SESSION_ID = f"MANUAL_TEST_{timestamp}"
    print(f"[*] Booting scenario. Data locked to: {SESSION_ID}")

    try:
        runner = ManualRunner(GAME_FILE, SESSION_ID)
        runner.run() 
    except KeyboardInterrupt:
        print("\nTesting aborted by user.")
    except Exception as e:
        print(f"Testing Error: {e}")