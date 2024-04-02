import pathlib
import pandas as pd
import numpy as np

from feature_definitions import load_data,df_10_predict,save_data

from sklearn.model_selection import train_test_split


# Importing dataframe

if __name__ == '__main__':
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    data_path=home_dir.as_posix() + '/data/raw/all_season_details.csv'
    df=load_data(data_path)
    df_10overruns=df_10_predict(df)
    print(df_10overruns.head())

    # Splitting the DataFrame into training (80%) and testing (20%) sets
    train_ddf_10overruns, test_df_10overruns = train_test_split(df_10overruns, test_size=0.2, random_state=2024)
    output_path = home_dir.as_posix() + '/data/processed'
    save_data(train_ddf_10overruns,test_df_10overruns,output_path)
