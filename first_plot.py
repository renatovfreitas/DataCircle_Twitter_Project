# Required libraries
import streamlit as st
import pandas as pd
import plotly.graph_objs as go  # Using Plotly

# Load dataset with caching
@st.cache_data
def load_data():
    # Updated file path
    return pd.read_csv(r"/Users/may/Documents/GitHub/DataCircle_Twitter_Project/twitter_cleaned_data.csv", lineterminator='\n')

# Load data
twitter_df = load_data()

# Clean column names (remove any extra newline characters)
twitter_df.columns = twitter_df.columns.str.replace('\r', '')

# Convert 'created_at' to datetime
twitter_df['created_at'] = pd.to_datetime(twitter_df['created_at'], errors='coerce')

# Create tabs
tab1, tab2, tab3 = st.tabs(["Exploratory Data Analysis", "Sentiment Analysis", "WordCloud and Emoji Analysis"])

# Visualizations in tab 1
with tab1:
    # Total tweet counts
    biden_tweets_count = twitter_df[twitter_df['candidate'] == 'biden']['candidate'].count()
    trump_tweets_count = twitter_df[twitter_df['candidate'] == 'trump']['candidate'].count()

    # Display tweet counts in larger font with spacing
    st.markdown(f"<h3 style='color:white;'>Biden's total tweets: {biden_tweets_count:,}</h3>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='color:white;'>Trump's total tweets: {trump_tweets_count:,}</h3>", unsafe_allow_html=True)

    # Group by date and candidate to get the count of tweets per day
    daily_tweets = twitter_df.groupby([twitter_df['created_at'].dt.date, 'candidate']).size().unstack(fill_value=0)

    # Create the plot (Line Chart)
    trace_biden = go.Scatter(
        x=daily_tweets.index, 
        y=daily_tweets['biden'], 
        mode='lines', 
        name='Biden Tweets',
        line=dict(color='blue', width=3)
    )
    
    trace_trump = go.Scatter(
        x=daily_tweets.index, 
        y=daily_tweets['trump'], 
        mode='lines', 
        name='Trump Tweets',
        line=dict(color='red', width=3)
    )

    # Add vertical lines for key dates (Election Day and Debate Day)
    election_day = go.layout.Shape(
        type='line',
        x0='2020-11-03',
        x1='2020-11-03',
        y0=0,
        y1=1,
        yref='paper',
        line=dict(color='yellow', dash='dash', width=2)
    )

    debate_day = go.layout.Shape(
        type='line',
        x0='2020-10-22',
        x1='2020-10-22',
        y0=0,
        y1=1,
        yref='paper',
        line=dict(color='cyan', dash='dash', width=2)
    )

    # Add annotations for the lines
    annotations = [
        dict(
            x='2020-11-03',
            y=1,
            xref='x',
            yref='paper',
            text='Election Day',
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-40,
            font=dict(color='yellow')
        ),
        dict(
            x='2020-10-22',
            y=1,
            xref='x',
            yref='paper',
            text='Debate Day',
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-40,
            font=dict(color='cyan')
        )
    ]

    # Define layout with black background
    layout = go.Layout(
        title=dict(
            text="Daily Tweets Count by Candidate",
            font=dict(size=20, color="white"),
            x=0.5,  # Center align the title
            y=0.95  # Adjust title position slightly higher
        ),
        xaxis=dict(
            title="Date",
            tickfont=dict(color='white')  # X-axis tick font color
        ),
        yaxis=dict(
            title="Number of Tweets",
            tickfont=dict(color='white')  # Y-axis tick font color
        ),
        shapes=[election_day, debate_day],
        annotations=annotations,  # Add annotations to the layout
        plot_bgcolor='black',  # Set plot background to black
        paper_bgcolor='black',  # Set overall background to black
        font=dict(family="Arial, sans-serif", size=12, color='white')  # Set font color to white for visibility
    )

    # Create the figure with the traces and layout
    fig = go.Figure(data=[trace_biden, trace_trump], layout=layout)

    # Display the plot in Streamlit
    st.plotly_chart(fig)

