from jericho import FrotzEnv

env = FrotzEnv("z-machine-games-master/jericho-game-suite/zork1.z5")
obs, info = env.reset()
print(obs)
from jericho import FrotzEnv

def main():
    env = FrotzEnv("z-machine-games-master/jericho-game-suite/zork1.z5")

    obs, info = env.reset()
    print("\n--- GAME START ---\n")
    print(obs)

    done = False
    step = 0

    while not done:
        print("\n-----------------")
        print(f"Step: {step}")

        try:
            actions = env.get_valid_actions()
            print("\nPossible actions:")
            possible_actions = {}
            for i, a in enumerate(actions[:20]):
                possible_actions[str(i)] = a
                print(f"{i}: {a}")
            print("...")
        except:
            actions = None

        # Ask user
        action = input("\nWhat should I do next? > ").strip()

        if action.lower() in ["quit", "exit"]:
            break
        try:
            current_action = possible_actions[action]
        except:
            print("Choice not available, defaulting to choice 0...")
            current_action = possible_actions['0']

        obs, reward, done, info = env.step(current_action)
        print(env.step("inventory"))
        print("\nObservation:")
        print(obs)

        print(f"\nReward: {reward}")
        print(f"Done: {done}")

        step += 1

    env.close()
    print("\n--- GAME OVER ---")

if __name__ == "__main__":
    main()

obs, reward, done, info = env.step("look")
print(obs)