import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_rewards_in_folder(root_folder):
    # Walk through the root folder and its subfolders
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.startswith('drl') and file.endswith('.csv'):
                file_path = os.path.join(subdir, file)
                
                # Read the CSV file
                df = pd.read_csv(file_path)   
                column_name = "reward/rew"

                # Use 10 step rolling window
                df[column_name] = df[column_name].rolling(window=100).mean()
                
                # Plot the column
                plt.figure(figsize=(15, 5))
                df[column_name].plot(title=f"Training Process")
                plt.xlabel('Training Step')
                plt.ylabel(column_name)

                # Plot between 0.70 and 0.80
                # plt.ylim(0.80, 1.00)

                # Plot x axis from 0 to 5000
                plt.xlim(50, 5000)
                
                # Save the plot in the same folder as the CSV
                plot_file_path = subdir + '/' + "reward" + '.png'
                print(plot_file_path)
                plt.savefig(plot_file_path, dpi=600, bbox_inches='tight')
                plt.close()
                print(f"Plot saved for {file} at {plot_file_path}")

# Specify the root folder
root_folder = 'fedLogsEval'
plot_rewards_in_folder(root_folder)