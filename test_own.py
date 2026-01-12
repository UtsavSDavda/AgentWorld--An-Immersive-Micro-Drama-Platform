from jericho import FrotzEnv

def main():
    env = FrotzEnv("npc1.z8")
    obs, info = env.reset()
    print("\n--- GAME START ---\n")
    print(obs)

    done = False
    step = 0

    while not done:
        action = input("\nCommand > ").strip()

        if action in ["quit", "exit"]:
            break

        obs, reward, done, info = env.step(action)

        print("\nObservation:")
        print(obs)
        print(f"\nReward: {reward}")
        print(f"Done: {done}")

    env.close()
    print("\n--- GAME OVER ---")

if __name__ == "__main__":
    main()