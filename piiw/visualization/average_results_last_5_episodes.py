import wandb
import numpy as np

# Function to get the average of the last 5 values of a specified metric
def get_average_of_last_n_points(runs, metric, n=5):
    averages = []
    for run in runs:
        # Retrieve the history of the specified metric
        history = run.history(keys=[metric])
        # Get the last n values of the metric
        last_n_values = history[metric].tail(n).values
        # Calculate the average and store it
        average = np.mean(last_n_values)
        averages.append(average)
    return np.mean(averages), averages


# Replace 'your_entity' and 'your_project' with your actual wandb entity and project name
api = wandb.Api()
entity = "piiw-thesis"
project = "pi-iw-experiments-piiw"

# Get runs from the project
runs = api.runs(f"{entity}/{project}")

# Filter runs by a specific group, if needed (optional)
group_names = [
    "group_Breakout-v4_2024-06-02_13-25-01.928315",
    "group_ChopperCommand-v4_2024-06-02_13-25-01.928297",
    "group_MsPacman-v4_2024-06-02_13-25-01.928277",
    "group_Pong-v4_2024-06-02_13-25-01.928230"
    ]# Set this to your specific group names if needed

for group_name in group_names:
    grouped_runs = [run for run in runs if run.group == group_name]

    # Calculate the average of the last 5 "train/rewards" for the grouped runs
    average_train_rewards, individual_averages = get_average_of_last_n_points(grouped_runs, "train/episode_reward")

    print(f"Results for group {group_name}")
    print(f"Average 'train/rewards' for the last 5 points across grouped runs: {average_train_rewards}")
    print(f"Individual averages: {individual_averages}")