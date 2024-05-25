import os
import pandas as pd
import matplotlib.pyplot as plt

def preprocess_csv(df):
    # Remove columns step, weightUpdates and episode
    df = df.drop(['step', 'weightUpdates', 'episode'], axis=1)

    # Remove every column starting with 'reward'
    df = df.drop([col for col in df if col.startswith('reward')], axis=1)

    # Remove brackets from every cell [1 2 3] -> 1, 2, 3
    df = df.replace('[\[\]]', '', regex=True)

    # For each cell, if the cell contains an array, replace it with the max value, unless the column is state_bandwidth, then use min value
    for col in df:
        for index, row in df.iterrows():
            if isinstance(row[col], str):
                # If the cell contains a comma, it is an array, split by comma
                if ',' in row[col]:
                    values = row[col].split(',')
                    if col == 'state_bandwidth':
                        df.at[index, col] = min(float(value) for value in values)
                    else:
                        df.at[index, col] = max(float(value) for value in values)

                elif ' ' in row[col]:
                    # If the cell contains a space, it is an array, split by space (may be multiple spaces)
                    values = row[col].split()
                    if col == 'state_bandwidth':
                        df.at[index, col] = min(float(value) for value in values)
                    else:
                        df.at[index, col] = max(float(value) for value in values)
    return df

def plot_rewards_in_folders(folder1, folder2, label1, label2):
    # Initialize lists to store data for plotting
    data_list = []
    labels = []

    # Function to process a single folder
    def process_folder(folder, label):
        for subdir, _, files in os.walk(folder):
            for file in files:
                if file.startswith('drl') and file.endswith('.csv'):
                    file_path = os.path.join(subdir, file)
                    
                    # Read the CSV file
                    df = pd.read_csv(file_path)   
                    df = preprocess_csv(df)
                    column_name = "state/fractionLossRate"


                    # Store data and labels for plotting
                    data_list.append(df[column_name])
                    labels.append(label)

    # Process both folders
    process_folder(folder1, label1)
    process_folder(folder2, label2)

    # Create subplots
    num_plots = len(data_list)
    if num_plots != 2:
        raise ValueError("The script requires exactly two folders with CSV files to plot.")

    fig, axs = plt.subplots(num_plots, 1, figsize=(15, 5 * num_plots))

    if num_plots == 1:
        axs = [axs]  # Ensure axs is iterable even if there's only one plot

    # Plot each dataframe in a subplot
    for i, data in enumerate(data_list):
        axs[i].plot(data)
        axs[i].set_title(f"Evaluation Run - {labels[i]}")
        axs[i].set_ylabel('Package Loss Rate')
        axs[i].set_xlim(0, 250)
        axs[i].set_ylim(0, 1)

        # Only set the x-axis label for the last subplot
        if i == num_plots - 1:
            axs[i].set_xlabel('Evaluation Step')

    # Save the combined plot
    plot_file_path = os.path.join(os.path.dirname(folder1), 'combined_package_loss_rate_' + label1 + '_' + label2 + '.png')
    plt.tight_layout()
    plt.savefig(plot_file_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Combined plot saved at {plot_file_path}")

# Specify the folders and their labels
folder1 = 'fedLogsRewardEval/fixed_gcc'
folder2 = 'fedLogsRewardEval/fixed_rRate = 0.25 (xxx)'
label1 = 'GCC'
label2 = 'Our Agent'

plot_rewards_in_folders(folder1, folder2, label1, label2)