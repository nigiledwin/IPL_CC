import streamlit as st
import pandas as pd
from joblib import load


# Define the input fields
st.write("# IPL Score Prediction App")

import pathlib
curr_dir = pathlib.Path(__file__)
home_dir = curr_dir.parent
data_path =  home_dir.as_posix() + '/models'
model_path = data_path + "/model.joblib"
model = load(model_path)

batting_team = st.selectbox("Select Batting Team", ["SRH", "MI", "CSK", "KKR", "RCB", "DC", "RR", "KXIP"])
bowling_team = st.selectbox("Select Bowling Team", ["SRH", "MI", "CSK", "KKR", "RCB", "DC", "RR", "KXIP"])
is_bat_home_team = st.radio("Is Batting Team Playing at Home?", ("Yes", "No"))
over = st.number_input("Over Number", min_value=1.0, max_value=20.0, value=6.0)
ball = st.number_input("Ball Number", min_value=1, value=6)
total_runs = st.number_input("Total Runs", min_value=0, value=70)
wickets = st.number_input("Wickets", min_value=0, value=1)

# Create a DataFrame from the user inputs
input_data = pd.DataFrame({
    'batting_team': [batting_team],
    'bowling_team': [bowling_team],
    'Is_bat_home_team': [is_bat_home_team],
    'over': [over],
    'ball': [ball],
    'total_runs': [total_runs],
    'Wkts': [wickets]
})

# Make predictions
prediction = model.predict(input_data)

# Display the prediction
st.write(f"Predicted Score: {prediction[0]}")




