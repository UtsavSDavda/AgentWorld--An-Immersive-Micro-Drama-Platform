import jericho
import sys

class ManualAgentController:
    def __init__(self, name):
        self.name = name

    def generate_action_catalog(self, context_data):
        """Builds a verified list of verbs based on the agent's immediate sight data."""
        catalog = ["wait"]
        if not context_data:
            return catalog

        # 1. Directions
        for direction in context_data.get("direction", []):
            catalog.append(f"go {direction}")

        # 2. People
        for person in context_data.get("person", []):
            catalog.append(f"examine {person}")

        # 3. Objects in room
        for obj, state in context_data.get("object", []):
            catalog.append(f"examine {obj}")
            catalog.append(f"take {obj}")
            catalog.extend(self._get_state_actions(obj, state))

        # 4. Inventory
        for item, state in context_data.get("inventory", []):
            catalog.append(f"examine {item}")
            catalog.append(f"drop {item}")
            catalog.extend(self._get_state_actions(item, state))

        # Remove duplicates and sort for a clean terminal menu
        return sorted(list(set(catalog)))

    def _get_state_actions(self, obj, state):
        """Maps Inform 7 physical states to available verbs."""
        actions = []
        if state == "device_off":
            actions.append(f"switch on {obj}")
        elif state == "device_on":
            actions.append(f"switch off {obj}")
        elif state == "openable_closed":
            actions.append(f"open {obj}")
        elif state == "openable_open":
            actions.append(f"close {obj}")
            actions.append(f"look inside {obj}")
        return actions


class JerichoRunner:
    def __init__(self, game_file):
        print(f"Loading {game_file}...")
        self.env = jericho.FrotzEnv(game_file)
        self.tick_count = 0
        self.agents = {} # Dynamically populated

    def parse_sight_data(self, observation_text):
        """Parses line-by-line to avoid regex newline issues."""
        sight_dict = {}
        
        for line in observation_text.splitlines():
            line = line.strip()
            if line.startswith("DATA_SIGHT:"):
                # Clean the string and split by the pipe delimiter
                clean_line = line.replace("DATA_SIGHT:", "").strip()
                parts = [p.strip() for p in clean_line.split("|")]
                
                if len(parts) >= 3:
                    observer = parts[0]
                    category = parts[1]
                    target = parts[2]
                    state = parts[3] if len(parts) >= 4 else "basic"
                    
                    # Register new agents dynamically
                    if observer not in sight_dict:
                        sight_dict[observer] = {"direction": [], "person": [], "object": [], "inventory": []}
                    if observer not in self.agents:
                        self.agents[observer] = ManualAgentController(observer)
                        
                    # Sort data into the dictionary
                    if category in ["object", "inventory"]:
                        sight_dict[observer][category].append((target, state))
                    else:
                        sight_dict[observer][category].append(target)
                        
        return sight_dict

    def parse_results(self, observation_text):
        """Extracts the SUCCESS/FAIL consequences of the last tick."""
        results = []
        for line in observation_text.splitlines():
            line = line.strip()
            if line.startswith("DATA_RESULT:"):
                clean_line = line.replace("DATA_RESULT:", "").strip()
                results.append(clean_line)
        return results

    def run(self):
        print("\n=== STARTING SIMULATION ===")
        # Initial dump to populate agents and first-turn sights
        obs, _, _, _ = self.env.step("dump sight")
        
        while True:
            self.tick_count += 1
            print(f"\n[{'='*15} TICK {self.tick_count} {'='*15}]")
            
            # 1. Parse current world state
            current_sight_data = self.parse_sight_data(obs)
            
            if not self.agents:
                print("No agents detected in the environment. Exiting.")
                break
                
            # 2. Prompt user for each agent's action
            for agent_name, agent_controller in self.agents.items():
                agent_context = current_sight_data.get(agent_name, {})
                catalog = agent_controller.generate_action_catalog(agent_context)
                
                print(f"\n--- {agent_name}'s Turn ---")
                print("Available Actions:")
                for idx, action in enumerate(catalog):
                    print(f"  {idx + 1}. {action}")
                
                # Get user input safely
                while True:
                    try:
                        choice = input(f"Choose action number for {agent_name} (or type 'q' to quit): ")
                        if choice.lower() == 'q':
                            print("Exiting simulation...")
                            sys.exit(0)
                            
                        choice_idx = int(choice) - 1
                        if 0 <= choice_idx < len(catalog):
                            chosen_action = catalog[choice_idx]
                            break
                        else:
                            print("Invalid number. Try again.")
                    except ValueError:
                        print("Please enter a valid number.")
                        
                # Inject the chosen plan silently via the backdoor
                self.env.step(f"force {agent_name} to {chosen_action}")
                print(f"> Injected plan for {agent_name}: {chosen_action}")

            # 3. Step the global simulation forward
            obs, _, _, _ = self.env.step("step")
            
            # 4. Display the narrative consequences
            print("\n--- Narrative Results ---")
            results = self.parse_results(obs)
            if results:
                for res in results:
                    print(f"Outcome: {res}")
            else:
                print("No specific agent actions logged this turn.")
                
            # 5. Fetch fresh sight data for the next loop
            obs, _, _, _ = self.env.step("dump sight")

if __name__ == "__main__":
    GAME_FILE = "games/Will5.z8"  # Replace with your compiled game file
    try:
        runner = JerichoRunner(GAME_FILE)
        runner.run()
    except Exception as e:
        print(f"Simulation Error: {e}")