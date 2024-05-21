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
                column_name = "action"

                # Add 1.0 to the column
                df[column_name] = df[column_name] + 1
                # Scale between 0 and 1
                df[column_name] = df[column_name] / 2

                
                # Plot the column
                plt.figure(figsize=(15, 5))
                df[column_name].plot(title=f"Training Run")
                plt.xlabel('Evaluation Step')
                plt.ylabel('bitrate (bps)')

                # Plot x axis from 0 to 250
                plt.xlim(0, 250)

                # Plot y axis from 0 to 1
                
                # Save the plot in the same folder as the CSV
                plot_file_path = subdir + '/' + "bitrate" + '.png'
                print(plot_file_path)
                plt.savefig(plot_file_path, dpi=600 , bbox_inches='tight')
                plt.close()
                print(f"Plot saved for {file} at {plot_file_path}")

# Specify the root folder
root_folder = 'fedLogsRewardEval'
plot_rewards_in_folder(root_folder)