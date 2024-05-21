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


def plot_in_folder(root_folder):
    # Walk through the root folder and its subfolders
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.startswith('drl') and file.endswith('.csv'):
                file_path = os.path.join(subdir, file)
                
                # Read the CSV file
                df = pd.read_csv(file_path)  
                df = preprocess_csv(df) 


                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

                # Plot bitrate
                bitrate = "action"
                df[bitrate] = df[bitrate] + 1
                # Scale between 0 and 1
                df[bitrate] = df[bitrate] / 2

                ax1.set_title("Bitrate")
                ax1.plot(df[bitrate], label=bitrate)
                ax1.set_ylabel('Bitrate')
                ax1.set_ylim(0.0, 1.0)
                ax1.set_xlim(0, 250)

        
                # Plot packet loss
                packet_loss = "state/fractionLossRate"

                ax2.set_title("Packet Loss")
                ax2.plot(df[packet_loss], label=packet_loss, color='red')
                ax2.set_ylabel('Packet Loss')
                ax2.set_ylim(0.0, 1.0)
                ax2.set_xlim(0, 250)

                plt.xlabel('Evaluation Step')


                # Save the plot in the same folder as the CSV
                plot_file_path = subdir + '/' + "both" + '.png'
                print(plot_file_path)

                plt.savefig(plot_file_path, dpi=600, bbox_inches='tight')
                plt.close()
                print(f"Plot saved for {file} at {plot_file_path}")



# Specify the root folder
root_folder = 'fedLogsRewardEval'
plot_in_folder(root_folder)