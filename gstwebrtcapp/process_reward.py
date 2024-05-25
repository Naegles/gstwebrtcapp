import os
import pandas as pd


def preprocess_csv(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

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

# Function to process a single CSV file
def process_df(df, folder_name):
    # Calculate the mean of each column
    means = df.mean()

    # Add the folder name as a new column
    means['Folder'] = folder_name
    processed_df =  pd.DataFrame(means.values.reshape(1, -1))
    return processed_df

def rename_columns(df):
    # Rename all columns
    df.columns = ['action', 'bandwidth', 'fractionLossRates', 'fractionNackrate', 
                  'fractionPliRate', 'fractionQueueingRtt', 'fractionRtt', 'interarrivalRttJitter', 'lossRate', 
                  'rttMean', 'rttStd', 'rxGoodput', 'txGoodput', 'Folder']
    
    # Make folder column the index
    df = df.set_index('Folder')
    return df

def finalize_results(df):
    df = df.apply(pd.to_numeric, errors='coerce')

    # Get the maximum value in each column
    max_values = df.max()

    # Replace the absolute values with deviation from the maximum in percentage
    for col in df:
        df[col] = df[col] / max_values[col] * 100

    # Round to 1 significant figures
    df = df.round(1)

    # Remove trailing zeros
    df = df.applymap(lambda x: '{:.1f}'.format(x))

    # Add row for actual maximum values
    df.loc['Max'] = max_values

    

    
    print(df)
    return df
    

def main():
    # Directory containing CSV files
    directory = 'fedLogsRewardEval'

    # Initialize an empty DataFrame to store processed results
    processed_results = pd.DataFrame()

    # Iterate through files and subdirectories
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith('drl') and file.endswith('.csv'):
                file_path = os.path.join(root, file)
                folder_name = os.path.basename(root)
                preprocessed_df = preprocess_csv(file_path)
                processed_df = process_df(preprocessed_df, folder_name)
                processed_results = pd.concat([processed_results, processed_df], axis=0)
    processed_results = rename_columns(processed_results)
    processed_results = finalize_results(processed_results)
    print(processed_results)
                

    # Save the processed results to a single CSV file
    processed_results.to_csv('processed_results_percentMax.csv', index=True)

if __name__ == "__main__":
    main()
