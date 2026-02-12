import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Load datasets
match = pd.read_csv('matches.csv')
delivery = pd.read_csv('deliveries.csv')

# Print info about the datasets
print("Matches dataset shape:", match.shape)
print("Deliveries dataset shape:", delivery.shape)

# Merge datasets on match_id (from deliveries) and id (from matches)
delivery_df = delivery.merge(match, left_on='match_id', right_on='id', how='left')

# Print columns to inspect the merged dataframe
print("Columns in merged dataframe:", delivery_df.columns)
print("Merged dataframe shape:", delivery_df.shape)

# Ensure 'total_runs' is numeric
delivery_df['total_runs'] = pd.to_numeric(delivery_df['total_runs'], errors='coerce')

# Handle missing or non-numeric values
delivery_df['total_runs'].fillna(0, inplace=True)

# Add a column for current score
delivery_df['current_score'] = delivery_df.groupby('match_id')['total_runs'].cumsum()

# Add columns for overs left and balls left
delivery_df['overs_left'] = 20 - delivery_df['over']
delivery_df['balls_left'] = delivery_df['overs_left'] * 6 - delivery_df['ball']

# Calculate wickets left
delivery_df['player_dismissed'] = delivery_df['player_dismissed'].fillna("0").apply(
    lambda x: x if x == "0" else "1"
).astype('int')
wickets_in_match = delivery_df.groupby('match_id')['player_dismissed'].cumsum()
delivery_df['wickets_left'] = 10 - wickets_in_match

# Calculate target score
delivery_df['target'] = delivery_df.groupby('match_id')['total_runs'].transform('sum')

# Calculate runs left
delivery_df['runs_left'] = delivery_df['target'] - delivery_df['current_score']

# Calculate current run rate (CRR)
delivery_df['current_run_rate'] = delivery_df['current_score'] * 6 / (120 - delivery_df['balls_left'])

# Calculate required run rate (RRR)
delivery_df['required_run_rate'] = delivery_df.apply(
    lambda row: row['runs_left'] * 6 / row['balls_left'] if row['balls_left'] > 0 else 0, 
    axis=1
)

# Print some statistics
print("Rows in original dataframe before filtering:", delivery_df.shape[0])

# Extract relevant columns
final_df = delivery_df[[
    'batting_team', 'bowling_team', 'city', 'runs_left', 'balls_left',
    'wickets_left', 'target', 'current_run_rate', 'required_run_rate', 'winner'
]]

# Inspect the first few rows
print(final_df.head())

# Now apply the filtering conditions and print the result
final_df = final_df[
    (delivery_df['inning'] == 2) &  # Only second innings
    (delivery_df['balls_left'] > 0) &  # Balls left
    (delivery_df['runs_left'] > 0)  # Runs left to chase
]

print("Rows after filtering:", final_df.shape[0])

# Encode target variable (1 for chasing team win, 0 for bowling team win)
final_df['result'] = final_df.apply(lambda row: 1 if row['batting_team'] == row['winner'] else 0, axis=1)

# Check if there is data to train the model
if final_df.shape[0] == 0:
    raise ValueError("No data available after filtering for training.")

# Separate features and target
X = final_df.drop(['result', 'winner'], axis=1)
y = final_df['result']

# Create a column transformer for categorical features
trf = ColumnTransformer([
    ('trf', OneHotEncoder(sparse=False, drop='first'), ['batting_team', 'bowling_team', 'city'])
], remainder='passthrough')

# Build the pipeline
pipe = Pipeline(steps=[
    ('step1', trf),
    ('step2', LogisticRegression(solver='liblinear'))
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the pipeline on training data
pipe.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipe.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.4f}')

# Save the pipeline
with open('pipe.pkl', 'wb') as f:
    pickle.dump(pipe, f)

print("Model training complete and pipeline saved as 'pipe.pkl'.")

