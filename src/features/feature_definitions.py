import pathlib
import pandas as pd
import numpy as np

# Importing dataframe
if __name__ == '__main__':
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent

def load_data(data_path):
    # Load your dataset from a given path
    df = pd.read_csv(data_path)
    return df

def df_10_predict(df):
    columns_to_keep = ['season', 'match_id', 'current_innings', 'innings_id', 'match_name', 'home_team', 'away_team', 'over', 'ball', 'runs', 'isBoundary', 'isWide', 'wicket_id', 'isNoball', 'batsman1_name', 'batsman_runs']
    df_filtered = df[columns_to_keep].copy()  # Ensure you're working on a copy

    # Adding additional columns
    df_filtered['total_runs'] = df.groupby(['match_id', 'innings_id'])['runs'].transform('cumsum').astype(int)
    df_filtered['Wkts'] = df_filtered.groupby(['match_id', 'innings_id'])['wicket_id'].transform(lambda x: x.notnull().cumsum()).astype(int)
    df_filtered['final_scores'] = df_filtered.groupby(['match_id', 'current_innings'])['runs'].transform('sum')
    df_filtered['1st_10overruns'] = df_filtered[df_filtered['over']<=10].groupby(['match_id', 'current_innings'])['runs'].transform('sum')
    df_filtered['1st_10overruns'] = df_filtered['1st_10overruns'].fillna(method='ffill')
    df_filtered['1st_10overwkts'] = df_filtered[df_filtered['over']<=10].groupby(['match_id', 'current_innings'])['wicket_id'].transform('count')
    df_filtered['1st_10overwkts'] = df_filtered['1st_10overwkts'].fillna(method='ffill')
    df_filtered['bowling_team'] = df_filtered.apply(lambda row: row['away_team'] if row['home_team'] == row['current_innings'] else row['home_team'], axis=1)
    df_filtered['Is_bat_home_team'] = df_filtered.apply(lambda row: 'Yes' if row['home_team'] == row['current_innings'] else 'No', axis=1)

    # Sort the DataFrame by season and match_id to ensure chronological order
    df_sorted = df_filtered.sort_values(by=['season', 'match_id'])

    # Group the DataFrame by the team
    grouped = df_sorted.groupby('current_innings')

    # Calculate the rolling mean of the final scores for the last 10 matches for each team
    rolling_mean = grouped['1st_10overruns'].rolling(window=10, min_periods=1).mean()

    # Reset the index to align the rolling mean values with the original DataFrame
    rolling_mean = rolling_mean.reset_index(level=0, drop=True)

    # Assign the rolling mean values to a new column in the original DataFrame
    df_filtered['last_10_matches_mean_10overscore'] = rolling_mean


    columns_to_select_ppruns=['current_innings','bowling_team','Is_bat_home_team','over', 'ball','total_runs','Wkts','1st_10overruns']
    df_pp_ml=df_filtered[columns_to_select_ppruns]
    df_pp_ml.rename(columns={'current_innings': 'batting_team'}, inplace=True)
    df_final=df_pp_ml[df_pp_ml['over']<=10]


    df_final.loc[df_final['batting_team'] == 'KXIP', 'batting_team'] = 'PBKS'
    df_final.loc[df_final['bowling_team'] == 'KXIP', 'bowling_team'] = 'PBKS'
    df_final.loc[df_final['batting_team'] == 'GL', 'batting_team'] = 'GT'
    df_final.loc[df_final['bowling_team'] == 'GL', 'bowling_team'] = 'GT'
    df_final.loc[df_final['batting_team'] == 'PWI', 'batting_team'] = 'RPS'
    df_final.loc[df_final['bowling_team'] == 'PWI', 'bowling_team'] = 'RPS'
    df_final = df_pp_ml[(df_pp_ml['1st_10overruns'] >= 15) & (df_pp_ml['1st_10overruns'] <= 150)]
    return df_final

#function to create features to predict final runs

def df_finalruns_predict(df):
    columns_to_keep = ['season', 'match_id', 'current_innings', 'innings_id', 'match_name', 'home_team', 'away_team', 'over', 'ball', 'runs', 'isBoundary', 'isWide', 'wicket_id', 'isNoball', 'batsman1_name', 'batsman_runs']
    df_filtered = df[columns_to_keep].copy()  # Ensure you're working on a copy

    # Adding additional columns
    df_filtered['total_runs'] = df.groupby(['match_id', 'innings_id'])['runs'].transform('cumsum').astype(int)
    df_filtered['Wkts'] = df_filtered.groupby(['match_id', 'innings_id'])['wicket_id'].transform(lambda x: x.notnull().cumsum()).astype(int)
    df_filtered['final_scores'] = df_filtered.groupby(['match_id', 'current_innings'])['runs'].transform('sum')
    df_filtered['last_30balsruns'] = df_filtered.groupby(['match_id', 'current_innings'])['runs'].transform('sum')
    df_filtered['last_30balswkts'] = df_filtered.groupby(['match_id', 'current_innings'])['wicket_id'].transform('count')
    df_filtered['bowling_team'] = df_filtered.apply(lambda row: row['away_team'] if row['home_team'] == row['current_innings'] else row['home_team'], axis=1)
    df_filtered['Is_bat_home_team'] = df_filtered.apply(lambda row: 'Yes' if row['home_team'] == row['current_innings'] else 'No', axis=1)
    
    # Sort the DataFrame by season and match_id to ensure chronological order
    df_sorted = df_filtered.sort_values(by=['season', 'match_id'])


    columns_to_select_finalruns=['current_innings','bowling_team','Is_bat_home_team','over', 'ball','total_runs','Wkts','final_scores']
    df_finalml=df_filtered[columns_to_select_finalruns]
    df_finalml.rename(columns={'current_innings': 'batting_team'}, inplace=True)
    


    df_finalml.loc[df_finalml['batting_team'] == 'KXIP', 'batting_team'] = 'PBKS'
    df_finalml.loc[df_finalml['bowling_team'] == 'KXIP', 'bowling_team'] = 'PBKS'
    df_finalml.loc[df_finalml['batting_team'] == 'GL', 'batting_team'] = 'GT'
    df_finalml.loc[df_finalml['bowling_team'] == 'GL', 'bowling_team'] = 'GT'
    df_finalml.loc[df_finalml['batting_team'] == 'PWI', 'batting_team'] = 'RPS'
    df_finalml.loc[df_finalml['bowling_team'] == 'PWI', 'bowling_team'] = 'RPS'
    return df_finalml


"""
Save Data Function:

This function saves two DataFrames (train and test) to CSV files with custom names.

Input Parameters:
- train: DataFrame containing the training data.
- test: DataFrame containing the testing data.
- train_output_name: Desired name for the CSV file containing the training data.
- test_output_name: Desired name for the CSV file containing the testing data.
- output_path: Directory where the CSV files will be saved.

Functionality:
1. Construct output file paths for the training and testing CSV files.
2. Save the train DataFrame to the specified output path with the provided name.
3. Save the test DataFrame to the specified output path with the provided name.

Description:
This function simplifies the process of saving data by providing a single function that handles
the entire process of constructing file paths and saving the DataFrames to CSV format.
"""
import os
def save_data(train, test, train_output_name, test_output_name, output_path):
    # Construct output file paths
    train_output_path = os.path.join(output_path, f"{train_output_name}.csv")
    test_output_path = os.path.join(output_path, f"{test_output_name}.csv")
    
    # Save the split datasets to the specified output paths
    train.to_csv(train_output_path, index=False)
    test.to_csv(test_output_path, index=False)
    
    # Save the split datasets to the specified output paths
    train.to_csv(train_output_path, index=False)
    test.to_csv(test_output_path, index=False)
