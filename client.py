import requests
import time
import sys

BASE_URL = "http://localhost:5000"

def run_test():
    print("🚀 --- Starting API Integration Test ---")

    # 1. Fetch available games
    print("\n[1] Fetching available games from server...")
    try:
        games_resp = requests.get(f"{BASE_URL}/games")
        games_resp.raise_for_status()
        games_list = games_resp.json().get("available_games", [])
        
        if not games_list:
            print("❌ No games found on the server. Please place a game file (e.g., .z8) in the 'games' folder.")
            sys.exit(1)
            
        print("✅ Found games:")
        for idx, game in enumerate(games_list):
            print(f"   [{idx}] {game}")

        while True:
            try:
                choice = int(input("Select a game by number: "))
                if 0 <= choice < len(games_list):
                    selected_game = games_list[choice]
                    break
                else:
                    print("Invalid selection. Try again.")
            except ValueError:
                print("Please enter a valid number.")
        
        # Dynamically select the first game in the list for the test
        selected_game = games_list[0]
        print(f"   Selecting '{selected_game}' for this test session.")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to fetch games. Is the server running? Error: {e}")
        sys.exit(1)

    # 2. Start the Game
    print(f"\n[2] Requesting to start game: {selected_game}...")
    try:
        response = requests.post(f"{BASE_URL}/game/start", json={"game_file": selected_game})
        response.raise_for_status()
        data = response.json()
        
        session_id = data.get("session_id")
        db_name = data.get("db_name")
        print(f"✅ Game started successfully!")
        print(f"   Session ID: {session_id}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to start game: {e}")
        sys.exit(1)

    time.sleep(1)

    # 3. Step through the game
    print("\n[3] Simulating 5 turns of gameplay...")
    for i in range(1, 6):
        print(f"   -> Executing Tick {i}...")
        try:
            step_resp = requests.post(f"{BASE_URL}/game/{session_id}/step")
            step_resp.raise_for_status()
            step_data = step_resp.json()
            
            tick = step_data.get("tick")
            locations = step_data.get("locations")
            print(f"      [Server Tick {tick}] Agent Locations: {locations}")
            time.sleep(1) 
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed on tick {i}: {e}")
            sys.exit(1)

    # 4. Trigger Video Generation
    print("\n[4] Requesting Video Generation for Ticks 0 through 5...")
    try:
        vid_payload = {"start_tick": 0, "end_tick": 5}
        vid_resp = requests.post(f"{BASE_URL}/game/{session_id}/generate_video", json=vid_payload)
        vid_resp.raise_for_status()
        
        print("✅ Video generation job accepted by server!")
        print("\n⏳ Check your Flask server terminal to watch the background thread process the video.")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to start video generation: {e}")

    print("\n🎉 --- Test Complete ---")

if __name__ == "__main__":
    run_test()