import os
import csv
from dotenv import load_dotenv
from chat_logger import SQLLogger

load_dotenv()

# ============================================================================
# EVAL 4: ACTION VALIDITY & COGNITIVE ADAPTATION
# ============================================================================

def evaluate_agent_learning(session_id: str, output_csv="agent_learning_metrics.csv"):
    print(f"\n🧠 Starting Cognitive Adaptation Eval for Session: {session_id}")
    db = SQLLogger(session_id)
    
    # Fetch all physical outcomes from the engine logs, chronologically
    response = db.supabase.table('engine_logs')\
        .select('tick, log_data')\
        .eq('session_id', session_id)\
        .eq('log_type', 'DATA_RESULT')\
        .order('tick').execute()
        
    engine_logs = response.data
    if not engine_logs:
        print("❌ No physical actions found in the engine logs for this session.")
        return

    # Tracking structures
    agent_stats = {}
    recent_failures = {} # Tracks the last failed action per agent to detect stubbornness
    
    # --- PHASE 1: PARSE AND TRACK ACTIONS ---
    for entry in engine_logs:
        tick = entry['tick']
        raw_log = entry['log_data'].replace("DATA_RESULT:", "").strip()
        parts = [p.strip() for p in raw_log.split("|")]
        
        if len(parts) >= 3:
            agent = parts[0]
            status = parts[1]
            action = parts[2]
            reason = parts[3] if len(parts) > 3 else "Unknown"
            
            # Initialize agent in our tracking dict
            if agent not in agent_stats:
                agent_stats[agent] = {
                    "total_actions": 0,
                    "successes": 0,
                    "first_time_failures": 0,
                    "stubborn_failures": 0,
                    "stubborn_log": [] # Keep track of what they failed to learn
                }
                
            agent_stats[agent]["total_actions"] += 1
            
            if status == "SUCCESS":
                agent_stats[agent]["successes"] += 1
                # If they succeeded, clear their recent failure memory 
                # (They did something productive, state changed)
                recent_failures[agent] = None 
                
            elif status == "FAIL":
                # Check for Cognitive Adaptation (Did they just fail doing this exact same thing?)
                last_fail = recent_failures.get(agent)
                
                if last_fail == action:
                    agent_stats[agent]["stubborn_failures"] += 1
                    agent_stats[agent]["stubborn_log"].append(f"Tick {tick}: Repeated '{action}' ({reason})")
                    print(f"  [🚨 STUBBORN] Tick {tick}: {agent} repeated failed action -> '{action}'")
                else:
                    agent_stats[agent]["first_time_failures"] += 1
                    recent_failures[agent] = action # Store this failure in their memory
                    print(f"  [⚠️ MISTAKE] Tick {tick}: {agent} failed first time -> '{action}' ({reason})")

    # --- PHASE 2: CALCULATE METRICS & EXPORT ---
    headers = ["session_id", "agent", "total_actions", "success_rate", "error_rate", "stubborn_rate"]
    file_exists = os.path.isfile(output_csv)
    
    with open(output_csv, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(headers)
            
        print("\n📊 --- COGNITIVE ADAPTATION REPORT ---")
        for agent, stats in agent_stats.items():
            total = stats["total_actions"]
            if total == 0:
                continue
                
            success_rate = (stats["successes"] / total) * 100
            error_rate = ((stats["first_time_failures"] + stats["stubborn_failures"]) / total) * 100
            
            # Stubborn rate is calculated out of total failures. 
            # "When you made a mistake, how often was it because you refused to learn?"
            total_fails = stats["first_time_failures"] + stats["stubborn_failures"]
            stubborn_rate = (stats["stubborn_failures"] / total_fails) * 100 if total_fails > 0 else 0.0

            writer.writerow([
                session_id, agent, total, 
                f"{success_rate:.1f}%", f"{error_rate:.1f}%", f"{stubborn_rate:.1f}%"
            ])
            
            print(f"\n👤 Agent: {agent}")
            print(f"   Total Physical Actions : {total}")
            print(f"   Valid Move Rate        : {success_rate:.1f}%")
            print(f"   Stubbornness Index     : {stubborn_rate:.1f}%")
            
            if stats["stubborn_log"]:
                print("   [Failure to Learn Log]:")
                for log in stats["stubborn_log"]:
                    print(f"      - {log}")

    print(f"\n✅ Learning metrics saved to {output_csv}")

if __name__ == "__main__":
    print("=========================================")
    print("🧠 AGENT COGNITIVE LEARNING EVALUATOR")
    print("=========================================")
    target_session = input("Enter the Session ID to evaluate: ").strip()
    
    if target_session:
        evaluate_agent_learning(target_session)
    else:
        print("❌ Session ID cannot be empty.")