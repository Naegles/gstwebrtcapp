import os
import pandas as pd


def preprocess_csv(file_path):
    df = pd.read_csv(file_path)

    # Remove columns step, weightUpdates and episode
    df = df.drop(['step', 'weightUpdates', 'episode'], axis=1)

    # Remove every column starting with 'reward'
    df = df.drop([col for col in df if col.startswith('reward')], axis=1)

    # Remove brackets from every cell [1 2 3] -> 1, 2, 3
    df = df.applymap(lambda x: x.replace('[', '').replace(']', ''))

    # For each cell, if the cell contains an array, replace it with the max value, unless the column is state_bandwidth, then use min value
    for col in df:
        for index, row in df.iterrows():
            if isinstance(row[col], str):
                if col == 'state_bandwidth':
                    df.at[index, col] = min([int(val) for val in row[col].split(',')])
                else:
                    df.at[index, col] = max([int(val) for val in row[col].split(',')])

    return df

# Function to process a single CSV file
def process_df(df):
    # Calculate the mean of each column
    means = df.mean()
    return means.values.reshape(1, -1)  # Reshape to a single row

def main():
    # Directory containing CSV files
    directory = 'fedLogsEval'

    # Initialize an empty DataFrame to store processed results
    processed_results = pd.DataFrame()

    # Iterate through files and subdirectories
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith('drl') and file.endswith('.csv'):
                file_path = os.path.join(root, file)
                folder_name = os.path.basename(root)
                df = preprocess_csv(file_path)
                processed_data = process_df(df)
                processed_df = pd.DataFrame(processed_data)  # Assuming df is available here
                processed_df.insert(0, 'Folder', folder_name)  # Insert folder name as the first column
                processed_results = pd.concat([processed_results, processed_df], axis=0)

    # Reset index to ensure correct indexing
    processed_results.reset_index(drop=True, inplace=True)

    # Save the processed results to a single CSV file
    processed_results.to_csv('processed_results.csv', index=False)

if __name__ == "__main__":
    main()
