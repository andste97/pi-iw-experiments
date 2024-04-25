import gym
import sys

from atari_utils.make_env import make_env
from utils.utils import display_image_cv2, transform_obs_to_image


def main(game_name):
    # Create the Atari environment
    #env = gym.make(game_name, render_mode='human')

    env = make_env(game_name, 18000, 15)

    # Reset the environment to start
    state = env.reset()
    #env.render()

    print("Use the keyboard to input actions.")
    print("Action space keys are:", env.unwrapped.get_action_meanings())

    # skip frames until game starts

    for i in range(5):
        observation, reward, done, info = env.step(0)

    img = transform_obs_to_image(observation)
    display_image_cv2("Observations", img, None, (800, 800))

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
        observation, reward, done, info = env.step(action)
        print(f"Reward: {reward}, Game Over: {done}, Info: {info}")

        img = transform_obs_to_image(observation)
        display_image_cv2("Observations", img, None,(800,800))

    env.close()

if __name__ == "__main__":
    #if len(sys.argv) < 2:
    #    print("Usage: python atari_interactive.py <AtariGameName>")
    #    sys.exit(1)

    #game_name = sys.argv[1] + "Pong-NoFrameskip-v4"
    game_name = "Pong-v4"
    main(game_name)