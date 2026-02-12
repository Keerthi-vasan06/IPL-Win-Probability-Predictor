import pandas as pd

# Load the matches CSV file
matches_df = pd.read_csv('matches.csv')

# Get the unique cities from the 'city' column
unique_cities = matches_df['team2'].unique()

# Print the unique cities
print("Unique cities in matches.csv:")
for city in unique_cities:
    print(city)
