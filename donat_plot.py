import streamlit as st
import pandas as pd
import plotly.graph_objs as go  # Using Plotly Graph Objects for Donut Charts

# Load dataset with caching
@st.cache_data
def load_data():
    # Updated file path
    return pd.read_csv(r"/Users/may/Documents/GitHub/DataCircle_Twitter_Project/twitter_sentiment.csv", lineterminator='\n')

# Load the data
twitter_df = load_data()

# Ensure all column names are lowercase and stripped of whitespace
twitter_df.columns = twitter_df.columns.str.strip().str.lower()

# Filter data for each candidate
biden_data = twitter_df[twitter_df['candidate'].str.strip().str.lower() == 'biden']
trump_data = twitter_df[twitter_df['candidate'].str.strip().str.lower() == 'trump']

# Check if 'sentiment' column exists
if 'sentiment' not in twitter_df.columns:
    st.error("The 'sentiment' column is missing from the dataset.")
else:
    # Group by sentiment for each candidate
    biden_sentiment_counts = biden_data['sentiment'].value_counts().reset_index()
    biden_sentiment_counts.columns = ['Sentiment', 'Count']

    trump_sentiment_counts = trump_data['sentiment'].value_counts().reset_index()
    trump_sentiment_counts.columns = ['Sentiment', 'Count']

    # Define shared colors for both charts
    shared_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green

    # Create combined figure with subplots
    fig = go.Figure()

    # Add Biden Donut Chart
    fig.add_trace(
        go.Pie(
            labels=biden_sentiment_counts['Sentiment'],
            values=biden_sentiment_counts['Count'],
            hole=0.4,
            name="Biden",
            marker=dict(colors=shared_colors),
            textinfo='percent+label',
            hoverinfo='label+percent+value',
            textfont=dict(size=18),
            textposition='inside',  # Center the text inside the donut
            pull=[0, 0, 0]  # Avoid any pull effect for the slices
        )
    )

    # Add Trump Donut Chart
    fig.add_trace(
        go.Pie(
            labels=trump_sentiment_counts['Sentiment'],
            values=trump_sentiment_counts['Count'],
            hole=0.4,
            name="Trump",
            marker=dict(colors=shared_colors),
            textinfo='percent+label',
            hoverinfo='label+percent+value',
            textfont=dict(size=18),
            textposition='inside',  # Center the text inside the donut
            pull=[0, 0, 0]  # Avoid any pull effect for the slices
        )
    )

    # Layout configuration
    fig.update_layout(
        title=dict(
            text="Sentiment Distribution for Biden and Trump",
            font=dict(color="white", size=24),
            x=0.5,  # Title aligned to the center
            xanchor='center'
        ),
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white"),
        showlegend=True,
        legend=dict(
            font=dict(size=16, color="white"),  # Larger font for legend
            bgcolor="black",
            bordercolor="white",
            borderwidth=1
        ),
        annotations=[
            dict(text="Biden", x=0.15, y=-0.15, font_size=30, showarrow=False, font_color="white"),  # Lower position for Biden
            dict(text="Trump", x=0.85, y=-0.15, font_size=30, showarrow=False, font_color="white")  # Lower position for Trump
        ]
    )

    # Adjust domains to make pie charts larger and centered
    fig.data[0].domain = {'x': [0.05, 0.45]}  # Biden Chart takes the left half
    fig.data[1].domain = {'x': [0.55, 0.95]}  # Trump Chart takes the right half

    # Display combined chart
    st.plotly_chart(fig, use_container_width=True)
