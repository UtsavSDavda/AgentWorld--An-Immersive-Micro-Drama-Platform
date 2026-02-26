import requests
import time
import sys

BASE_URL = "http://localhost:5000"
GAME_FILE = "Control4.z8"  # Ensure this file is in the server's working directory

def run_test():
    print("🚀 --- Starting API Integration Test ---")

    # 1. Start the Game
    print(f"\n[1] Requesting to start game: {GAME_FILE}...")
    try:
        response = requests.post(f"{BASE_URL}/game/start", json={"game_file": GAME_FILE})
        response.raise_for_status()
        data = response.json()
        
        session_id = data.get("session_id")
        db_name = data.get("db_name")
        print(f"✅ Game started successfully!")
        print(f"   Session ID: {session_id}")
        print(f"   Database: {db_name}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to start game. Is the Flask server running? Error: {e}")
        sys.exit(1)

    time.sleep(1)

    # 2. Step through the game (Simulating 5 turns)
    print("\n[2] Simulating 5 turns of gameplay...")
    for i in range(1,3):
        print(f"   -> Executing Tick {i}...")
        try:
            step_resp = requests.post(f"{BASE_URL}/game/{session_id}/step")
            step_resp.raise_for_status()
            step_data = step_resp.json()
            
            tick = step_data.get("tick")
            locations = step_data.get("locations")
            print(f"      [Server Tick {tick}] Agent Locations: {locations}")
            
            # Pause slightly to mimic processing/reading time
            time.sleep(1) 
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed on tick {i}: {e}")
            sys.exit(1)

    # 3. Trigger Video Generation
    print("\n[3] Requesting Video Generation for Ticks 1 through 2...")
    try:
        vid_payload = {"start_tick": 0, "end_tick": 5}
        vid_resp = requests.post(f"{BASE_URL}/game/{session_id}/generate_video", json=vid_payload)
        vid_resp.raise_for_status()
        
        print("✅ Video generation job accepted by server!")
        print(f"   Server Response: {vid_resp.json().get('message')}")
        print("\n⏳ Note: Check your Flask server terminal to watch the background thread process the video.")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to start video generation: {e}")

    print("\n🎉 --- Test Complete ---")

if __name__ == "__main__":
    run_test()