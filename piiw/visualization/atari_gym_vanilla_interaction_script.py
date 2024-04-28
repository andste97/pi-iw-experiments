import gym
import sys

def main(game_name):
    # Create the Atari environment
    env = gym.make(game_name, render_mode='human')

    # Reset the environment to start
    state = env.reset()
    env.render()

    print("Use the keyboard to input actions.")
    print("Action space keys are:", env.unwrapped.get_action_meanings())

    # Continue until the game is done
    done = False
    while not done:
        # Get user input for action
        action = input("Enter an action (integer): ")

        # Check if the input is an integer and a valid action
        try:
            action = int(action)
            assert action in range(env.action_space.n)
        except (ValueError, AssertionError):
            print("Invalid action. Please enter a valid action integer.")
            continue

        # Step the environment with the chosen action
        observations, reward, termination, truncation, info = env.step(action)
        env.render()

        print(f"Reward: {reward}, Game Over: {done}, Info: {info}")

    env.close()

if __name__ == "__main__":
    #if len(sys.argv) < 2:
    #    print("Usage: python atari_interactive.py <AtariGameName>")
    #    sys.exit(1)

    #game_name = sys.argv[1] + "Pong-NoFrameskip-v4"
    game_name = "PongDeterministic-v4"
    main(game_name)