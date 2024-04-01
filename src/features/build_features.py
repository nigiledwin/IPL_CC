import pathlib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent

data_path=home_dir.as_posix() + '/data/raw/all_season_details.csv'
def load_data(data_path):
    # Load your dataset from a given path
    df = pd.read_csv(data_path)
    return df

df=load_data(data_path)