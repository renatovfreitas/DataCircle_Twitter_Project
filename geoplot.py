import pandas as pd
import plotly.express as px
import os
import streamlit as st

# Title of the app
st.title("Twitter Analysis: US States Tweet Count by Candidate")

# Path to your CSV file
file_path = '/Users/may/Documents/GitHub/DataCircle_Twitter_Project/twitter_cleaned_data.csv'

# State abbreviation dictionary for conversion
state_abbreviations = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
    'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA',
    'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
    'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA',
    'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT',
    'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM',
    'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK',
    'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD',
    'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA',
    'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'
}

if os.path.exists(file_path):
    # Load dataset
    df_combined = pd.read_csv(file_path, lineterminator='\n')

    # Filter tweets for U.S.
    us_tweets = df_combined[df_combined['country'].str.strip().str.lower() == 'united states']

    # Ensure states have consistent capitalization and convert full names to abbreviations
    us_tweets['state'] = us_tweets['state'].str.title()
    us_tweets['state'] = us_tweets['state'].map(state_abbreviations)  # Convert to abbreviations

    # Drop any rows where state conversion failed (NaN in 'state' column)
    us_tweets = us_tweets.dropna(subset=['state'])

    # Group by state and candidate to calculate tweet counts
    tweets_by_state_and_candidate = us_tweets.groupby(['state', 'candidate']).size().reset_index(name='Tweet Count')

    # Display the data to check it’s being counted correctly
    st.write("Tweet Count by State and Candidate:")
    st.dataframe(tweets_by_state_and_candidate)

    # Add a dropdown filter for candidate selection
    candidate = st.selectbox("Select a candidate:", options=['Trump', 'Biden'])

    # Filter data for the selected candidate
    filtered_data = tweets_by_state_and_candidate[tweets_by_state_and_candidate['candidate'].str.strip().str.lower() == candidate.lower()]

    # Create a choropleth map for tweet counts by state for the selected candidate
    fig = px.choropleth(
        filtered_data,
        locations='state',
        locationmode="USA-states",
        color='Tweet Count',
        color_continuous_scale="Viridis",
        scope="usa",
        title=f'Tweet Count by US State for {candidate}',
        labels={'state': 'State', 'Tweet Count': 'Number of Tweets'}
    )

    # Display the map
    st.plotly_chart(fig)

else:
    st.write("File not found.")
