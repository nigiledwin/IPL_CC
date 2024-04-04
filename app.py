import streamlit as st
import pandas as pd
from joblib import load

# Define page configurations
st.set_page_config(page_title="IPL Score Prediction App", page_icon=":cricket:", layout="wide")
# Define the input fields
def ipl_10_over_score_prediction():
    st.write("# IPL Score Prediction App")

    import pathlib
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent
    data_path =  home_dir.as_posix() + '/models'
    model_path = data_path + "/model_xgb_10oversruns_.joblib"
    model = load(model_path)

    # Define the sidebar for input fields
    st.sidebar.title("IPL Score Prediction")
    batting_team = st.sidebar.selectbox("Select Batting Team", ["SRH", "MI", "CSK", "KKR", "RCB", "DC", "RR", "KXIP"])
    bowling_team = st.sidebar.selectbox("Select Bowling Team", ["MI", "SRH", "CSK", "KKR", "RCB", "DC", "RR", "KXIP"])
    is_bat_home_team = st.sidebar.radio("Is Batting Team Playing at Home?", ("Yes", "No"))
    over = st.sidebar.number_input("Over Number", min_value=1.0, max_value=20.0, value=6.0)
    ball = st.sidebar.number_input("Ball Number", min_value=1, value=6)
    total_runs = st.sidebar.number_input("Total Runs", min_value=0, value=70)
    wickets = st.sidebar.number_input("Wickets", min_value=0, value=1)

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
    if st.sidebar.button("Predict"):
        # Make predictions
        prediction = model.predict(input_data)

        # Display the prediction
        st.write(f"Predicted Score: {prediction[0]}")
def final_runs_prediction():
    st.write('need to add')


# Create navigation menu
app_mode = st.sidebar.selectbox("Select App Mode", ["IPL first 10 Score Prediction", "Final Runs Prediction"])

# Render the selected page
if app_mode == "IPL first 10 Score Prediction":
    ipl_10_over_score_prediction()
else:
    final_runs_prediction()        

'''

#fast api

# main.py
import pandas as pd
from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel

app = FastAPI()

class PredictionInput(BaseModel):
    # Define the input parameters required for making predictions
        batting_team: str
        bowling_team:str
        Is_bat_home_team:bool
        over:int
        ball:int
        total_runs:int
        Wkts:int
        

# Load the pre-trained RandomForest model
import pathlib
curr_dir = pathlib.Path(__file__)
home_dir = curr_dir.parent
data_path =  home_dir.as_posix() + '/models'
model_path = data_path + "/model.joblib"
model = load(model_path)

@app.get("/")
def home():
    return "Working fine"

@app.post("/predict")
def predict(input_data: PredictionInput):
    # Extract features from input_data and make predictions using the loaded model
    features = {
            'batting_team': input_data.batting_team,
            'bowling_team': input_data.bowling_team,
            'Is_bat_home_team': input_data.Is_bat_home_team,
            'over': input_data.over,
            'ball': input_data.ball,
            'total_runs': input_data.total_runs,
            'Wkts': input_data.Wkts,
            
}
    features = pd.DataFrame(features, index=[0])
    prediction = model.predict(features)[0].item()
    # Return the prediction
    print(prediction)
    return {"prediction": prediction}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)
'''