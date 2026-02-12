import streamlit as st
import pandas as pd
import pickle

# Load the trained pipeline
with open('pipe.pkl', 'rb') as f:
    pipe = pickle.load(f)

# Title for the Streamlit app
st.title('IPL Match Winner Prediction')

# List of IPL teams (use actual teams from your dataset)
teams = [
    'Royal Challengers Bangalore', 'Rising Pune Supergiant', 'Kolkata Knight Riders',
    'Kings XI Punjab', 'Delhi Daredevils', 'Sunrisers Hyderabad', 'Mumbai Indians',
    'Gujarat Lions', 'Rajasthan Royals', 'Chennai Super Kings', 'Deccan Chargers',
    'Pune Warriors', 'Kochi Tuskers Kerala', 'Rising Pune Supergiants', 'Delhi Capitals'
]

# Input form to take user input
with st.form("prediction_form"):
    # Input fields for user
    batting_team = st.selectbox('Batting Team', teams)
    bowling_team = st.selectbox('Bowling Team', teams)
    city = st.selectbox('Venue (City)', [
    'Hyderabad', 'Pune', 'Rajkot', 'Indore', 'Bangalore', 'Mumbai', 'Kolkata',
    'Delhi', 'Chandigarh', 'Kanpur', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
    'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley', 'Bloemfontein',
    'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala', 'Kochi', 'Visakhapatnam', 'Raipur',
    'Ranchi', 'Abu Dhabi', 'Sharjah', 'Mohali', 'Bengaluru'
])

    current_score = st.number_input('Current Score', min_value=0)
    wickets_out = st.number_input('Wickets Out', min_value=0)
    overs_completed = st.number_input('Overs Completed', min_value=0)
    target = st.number_input('Target', min_value=0)
    submit_button = st.form_submit_button("Predict")

if submit_button:
    # Perform necessary calculations
    runs_left = target - current_score
    balls_left = (20 - overs_completed) * 6  # Assuming 20 overs per side
    wickets_left = 10 - wickets_out
    current_run_rate = current_score / (overs_completed if overs_completed != 0 else 1)
    required_run_rate = runs_left / (balls_left / 6 if balls_left != 0 else 1)

    # Create input DataFrame
    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets_left': [wickets_left],
        'target': [target],
        'current_run_rate': [current_run_rate],
        'required_run_rate': [required_run_rate]
    })

    # Ensure the column names match exactly the ones used during training
    expected_columns = ['batting_team', 'bowling_team', 'city', 'runs_left', 'balls_left', 'wickets_left', 'target', 'current_run_rate', 'required_run_rate']
    
    # Check if the columns in input_df are the same as expected
    missing_columns = [col for col in expected_columns if col not in input_df.columns]
    if missing_columns:
        for col in missing_columns:
            input_df[col] = 0  # Fill missing columns with default value (0 or appropriate default)
    
    # Reorder the columns to match the training pipeline
    input_df = input_df[expected_columns]

    # Handle categorical data (ensure teams and city match known categories)
    known_teams = teams  # Replace with actual teams
    known_cities = [
        'Hyderabad', 'Pune', 'Rajkot', 'Indore', 'Bangalore', 'Mumbai', 'Kolkata', 'Delhi',
        'Chandigarh', 'Kanpur', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth', 'Durban',
        'Centurion', 'East London', 'Johannesburg', 'Kimberley', 'Bloemfontein', 'Ahmedabad',
        'Cuttack', 'Nagpur', 'Dharamsala', 'Kochi', 'Visakhapatnam', 'Raipur', 'Ranchi', 
        'Abu Dhabi', 'Sharjah', 'Mohali', 'Bengaluru'
    ]
    input_df['batting_team'] = input_df['batting_team'].apply(lambda x: x if x in known_teams else 'Mumbai Indians')  # Default to 'Mumbai Indians'
    input_df['bowling_team'] = input_df['bowling_team'].apply(lambda x: x if x in known_teams else 'Mumbai Indians')  # Default to 'Mumbai Indians'
    input_df['city'] = input_df['city'].apply(lambda x: x if x in known_cities else 'Mumbai')  # Default to 'Mumbai'

    # Predict the probability of the batting team winning
    result = pipe.predict_proba(input_df)

    # Display the result
    st.write(f"Probability of Batting Team Winning: {result[0][1]:.2f}")
    st.write(f"Probability of Bowling Team Winning: {result[0][0]:.2f}")