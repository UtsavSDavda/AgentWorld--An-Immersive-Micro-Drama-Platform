import sqlite3
from jericho import FrotzEnv
from datetime import datetime
import os

class ChatDatabase:
    def __init__(self, db_name="simple_chat.db"):
        if os.path.exists(db_name):
            os.remove(db_name)
            
        self.conn = sqlite3.connect(db_name)
        self.create_table()
        print(f"DEBUG: Connected to database '{db_name}'")

    def create_table(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                sender TEXT,
                receiver TEXT,
                message TEXT
            )
        ''')
        self.conn.commit()

    def save_log(self, sender, receiver, message):
        print(f"DEBUG: Entered save_log function. Data: {sender} -> {receiver}")
        
        cursor = self.conn.cursor()
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        try:
            cursor.execute('''
                INSERT INTO logs (timestamp, sender, receiver, message)
                VALUES (?, ?, ?, ?)
            ''', (timestamp, sender, receiver, message))
            
            self.conn.commit()
            print("DEBUG: SQL Commit Successful.")
        except Exception as e:
            print(f"ERROR: SQL Failed: {e}")

    def show_history(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM logs")
        rows = cursor.fetchall()
        print(f"\n--- DB CONTENTS ({len(rows)} entries) ---")
        for row in rows:
            print(row)
        print("---------------------------------------")

def trigger_chat(db, sender, receiver, message):
    print(f"\n[SYSTEM] Triggering chat: {sender} says '{message}' to {receiver}")
    db.save_log(sender, receiver, message)

def main():
    # Setup DB
    db = ChatDatabase()
    
    # Setup Game
    try:
        env = FrotzEnv("ColdIronNew.z8") 
        obs, _ = env.reset()
        print("Game Loaded Successfully.")
    except Exception as e:
        print(f"Game Load Error: {e}")
        return

    print("\nCOMMANDS:")
    print("1. 'chat [name] [message]' -> Sends a message from YOU to [name]")
    print("2. 'debug [sender] [receiver] [message]' -> Forces a chat between two NPCs")
    print("3. 'history' -> View DB logs")
    print("4. Any other key -> Played as a game command")

    while True:
        user_input = input("\n> ").strip()
        
        if user_input == "quit": 
            break

        elif user_input.lower().startswith("chat "):
            try:
                parts = user_input.split(" ", 2) # Split into max 3 parts
                if len(parts) < 3:
                    print("Error: Use format 'chat [Name] [Message]'")
                    continue
                    
                target = parts[1]
                msg = parts[2]
                trigger_chat(db, "Player", target, msg)
                
                # Keep game moving
                env.step("z") 

            except Exception as e:
                print(f"Parsing Error: {e}")

        elif user_input.lower().startswith("debug "):
            try:
                parts = user_input.split(" ", 3)
                if len(parts) < 4:
                    print("Error: Use format 'debug [Sender] [Receiver] [Message]'")
                    continue
                
                sender = parts[1]
                receiver = parts[2]
                msg = parts[3]
                
                trigger_chat(db, sender, receiver, msg)
                
            except Exception as e:
                print(f"Parsing Error: {e}")

        elif user_input == "history":
            db.show_history()

        else:
            obs, _, done, _ = env.step(user_input)
            print(obs)
            if done: break

if __name__ == "__main__":
    main()