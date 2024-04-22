import os
import csv
import pandas as pd

# Function to calculate average of floats in a column
def calculate_average(csv_file):
    df = pd.read_csv(csv_file)
    avg_reward = df['episode_rewards'].mean()
    return avg_reward

# Function to search for CSV files and calculate average rewards
def process_folder(folder_path):
    avg_rewards = []
    folder_name = os.path.basename(folder_path)
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv") and file.startswith("sac_eval_output"):
                csv_file = os.path.join(root, file)
                avg_reward = calculate_average(csv_file)
                avg_rewards.append((folder_name, avg_reward))
    return avg_rewards

# Function to save results to a CSV file
def save_to_csv(results, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['Folder Name', 'Average Reward']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({'Folder Name': result[0], 'Average Reward': result[1]})

# Main function
def main():
    folder_path = "fedLogsEval"
    output_file = "average_rewards.csv"
    all_avg_rewards = []
    for entry in os.scandir(folder_path):
        if entry.is_dir():
            avg_rewards = process_folder(entry.path)
            all_avg_rewards.extend(avg_rewards)
    save_to_csv(all_avg_rewards, output_file)
    print("Average rewards calculated and saved to", output_file)

if __name__ == "__main__":
    main()
