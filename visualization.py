# Required libraries
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import io
from datetime import datetime
from wordcloud import WordCloud

# Load dataset with caching
@st.cache_data
def load_data():
    return pd.read_csv(r"C:\Users\User\iCloudDrive\Cursos\Data Circle\DataCircle_Twitter_Project\twitter_sentiment.csv", lineterminator='\n')

# Load data
twitter_df = load_data()

# Clean column names
twitter_df.columns = twitter_df.columns.str.replace('\r', '')

# Convert created_at to datetime
twitter_df['created_at'] = pd.to_datetime(twitter_df['created_at'], errors='coerce')

# Create tabs
tab1, tab2, tab3 = st.tabs(["Exploratory Data Analysis", "Sentiment Analysis", "WordCloud and Emoji Analysis"])

# Visualizations in tab 1
with tab1:
    st.write(r"Biden's total tweets:", twitter_df[twitter_df['candidate']=='biden']['candidate'].count())
    st.write(r"Trump's total tweets:", twitter_df[twitter_df['candidate']=='trump']['candidate'].count())

    # Visualization 1: Total Tweets by Candidate
    
    # Plotting
    st.header("Total Tweets per Day by Candidate")

    # Group by date and candidate to get the count of tweets per day
    daily_tweets = twitter_df.groupby([twitter_df['created_at'].dt.date, 'candidate']).size().unstack()

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    daily_tweets.plot(kind='bar', ax=ax, color=['blue', 'red'], width=0.8)

    # Set titles and labels
    ax.set_title("Daily Tweets Count by Candidate")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Tweets")
    plt.xticks(rotation=45)

    # Display the plot in Streamlit
    st.pyplot(fig)

    # Visualization 2: Tweet Engagement (Likes and Retweets)

    # Plotting
    st.header("Tweet Engagement (Likes and Retweets)")

    # Group by candidate to get total likes and retweets
    engagement_data = twitter_df.groupby('candidate').agg({'likes': 'sum', 'retweet_count': 'sum'}).reset_index()

    # Create the plot with Plotly Express
    fig = px.bar(
        engagement_data,
        x='candidate',
        y=['likes', 'retweet_count'],
        barmode='group',
        title="Total Likes and Retweets by Candidate",
        labels={'value': 'Count', 'variable': 'Engagement Type'},
        color_discrete_sequence=['lightgreen', 'darkgreen']
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig)


# Visualization 3: Geoplot

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

    # Filter tweets for U.S.
    us_tweets = twitter_df[twitter_df['country'].str.strip().str.lower() == 'united states']

    # Ensure states have consistent capitalization and convert full names to abbreviations
    us_tweets['state'] = us_tweets['state'].str.title()
    us_tweets['state'] = us_tweets['state'].map(state_abbreviations)  # Convert to abbreviations

    # Drop any rows where state conversion failed (NaN in 'state' column)
    us_tweets = us_tweets.dropna(subset=['state'])

    # Group by state and candidate to calculate tweet counts
    tweets_by_state_and_candidate = us_tweets.groupby(['state', 'candidate']).size().reset_index(name='Tweet Count')

    st.header("Tweets by States")

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


# Visualizations in tab 2
with tab2:
    # Visualization 4: Sentiment Analysis

    # Create pie charts
    st.header("Sentiment Analysis by Candidate")

    # Filter data for each candidate
    biden_data = twitter_df[twitter_df['candidate'] == 'biden']
    trump_data = twitter_df[twitter_df['candidate'] == 'trump']

    # Group by sentiment for each candidate
    biden_sentiment_counts = biden_data['sentiment'].value_counts().reset_index()
    biden_sentiment_counts.columns = ['Sentiment', 'Count']

    trump_sentiment_counts = trump_data['sentiment'].value_counts().reset_index()
    trump_sentiment_counts.columns = ['Sentiment', 'Count']

    # Plot for Biden
    fig_biden = px.pie(
        biden_sentiment_counts,
        values='Count',
        names='Sentiment',
        title='Biden Sentiment Distribution',
        color_discrete_sequence=['#ffeb3b', '#4caf50', '#f44336']  # Colors: yellow, green, red
    )

    # Plot for Trump
    fig_trump = px.pie(
        trump_sentiment_counts,
        values='Count',
        names='Sentiment',
        title='Trump Sentiment Distribution',
        color_discrete_sequence=['#ffeb3b', '#4caf50', '#f44336']  # Colors: yellow, green, red
    )

    # Display the plots side by side
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_biden, use_container_width=True)
    with col2:
        st.plotly_chart(fig_trump, use_container_width=True)


    # # Visualization 5: Sentiment Trends Over Time 

    # Group by date and sentiment for Biden
    biden_sentiment_trends = biden_data.groupby([pd.Grouper(key='created_at', freq='D'), 'sentiment']).size().unstack(fill_value=0)
    # Group by date and sentiment for Trump
    trump_sentiment_trends = trump_data.groupby([pd.Grouper(key='created_at', freq='D'), 'sentiment']).size().unstack(fill_value=0)

    # Plot for Biden
    fig_biden = go.Figure()
    for sentiment in biden_sentiment_trends.columns:
        fig_biden.add_trace(go.Scatter(
            x=biden_sentiment_trends.index,
            y=biden_sentiment_trends[sentiment],
            mode='lines',
            name=sentiment
        ))

    # Plot for Biden
    fig_trump = go.Figure()
    for sentiment in trump_sentiment_trends.columns:
        fig_trump.add_trace(go.Scatter(
            x=trump_sentiment_trends.index,
            y=trump_sentiment_trends[sentiment],
            mode='lines',
            name=sentiment
        ))

    # Add vertical line for election day
    election_day = datetime(2020, 11, 3)
    fig_biden.add_vline(x=election_day, line_width=2, line_dash="dash", line_color="white")
    fig_trump.add_vline(x=election_day, line_width=2, line_dash="dash", line_color="white")

    # Add annotation for election day
    fig_biden.add_annotation(
        x=election_day,
        y=max(biden_sentiment_trends.max()),
        text="Election Day",
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-40,
        font=dict(color="white")
    )

    fig_trump.add_annotation(
        x=election_day,
        y=max(trump_sentiment_trends.max()),
        text="Election Day",
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-40,
        font=dict(color="white")
    )

    fig_biden.update_layout(title=f'Sentiment Trends Over Time for Biden', xaxis_title='Date', yaxis_title='Number of Tweets')
    fig_trump.update_layout(title=f'Sentiment Trends Over Time for Trump', xaxis_title='Date', yaxis_title='Number of Tweets')
    
    # Display the plots side by side
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_biden, use_container_width=True)
    with col2:
        st.plotly_chart(fig_trump, use_container_width=True)


# Visualization 6: Word Clouds

with tab3:
    # Function to generate word cloud from CSV
    def generate_wordcloud(csv_file):
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Create a dictionary from the DataFrame
        word_freq = dict(zip(df['Word'], df['Frequency']))
        
        # Generate the word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
        
        return wordcloud


    # Generate word clouds for both candidates
    biden_positive_wordcloud = generate_wordcloud('biden_positive_wordcloud.csv')
    biden_negative_wordcloud = generate_wordcloud('biden_negative_wordcloud.csv')
    trump_positive_wordcloud = generate_wordcloud('trump_positive_wordcloud.csv')
    trump_negative_wordcloud = generate_wordcloud('trump_negative_wordcloud.csv')

    # Create side-by-side Positive word clouds
    positive_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # Display Biden's word cloud
    ax1.imshow(biden_positive_wordcloud, interpolation='bilinear')
    ax1.axis('off')  # Hide axes
    ax1.set_title("Biden's Positive Word Cloud", fontsize=16)

    # Display Trump's word cloud
    ax2.imshow(trump_positive_wordcloud, interpolation='bilinear')
    ax2.axis('off')  # Hide axes
    ax2.set_title("Trump's Positive Word Cloud", fontsize=16)

    # Show the plot in Streamlit
    st.pyplot(positive_fig)

    # Create side-by-side Negative word clouds
    negative_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # Display Biden's word cloud
    ax1.imshow(biden_negative_wordcloud, interpolation='bilinear')
    ax1.axis('off')  # Hide axes
    ax1.set_title("Biden's Negative Word Cloud", fontsize=16)

    # Display Trump's word cloud
    ax2.imshow(trump_negative_wordcloud, interpolation='bilinear')
    ax2.axis('off')  # Hide axes
    ax2.set_title("Trump's Negative Word Cloud", fontsize=16)

    # Show the plot in Streamlit
    st.pyplot(negative_fig)

# Visualization 7: Emoji Analysis

    st.header("Emoji Analysis")
    
    # Function to load emoji data from CSV and get top 10 emojis
    def load_top_emojis(csv_file):
        df = pd.read_csv(csv_file)
        top_emojis = df.nlargest(5, 'Frequency')  # Get top 10 emojis
        return top_emojis

    # Streamlit application
    st.title("Top 5 Emojis for Biden and Trump")

    # Load top emojis for both candidates
    biden_emojis = load_top_emojis('biden_emojis.csv')
    trump_emojis = load_top_emojis('trump_emojis.csv')

    # Create individual bar charts for each candidate
    biden_fig = px.bar(biden_emojis, 
                    x='Emoji', 
                    y='Frequency', 
                    title='Top 10 Emojis for Biden',
                    labels={'Emoji': 'Emoji', 'Frequency': 'Frequency'},
                    color_discrete_sequence=['blue'])

    trump_fig = px.bar(trump_emojis, 
                    x='Emoji', 
                    y='Frequency', 
                    title='Top 10 Emojis for Trump',
                    labels={'Emoji': 'Emoji', 'Frequency': 'Frequency'},
                    color_discrete_sequence=['red'])

    # Display charts side by side
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(biden_fig)

    with col2:
        st.plotly_chart(trump_fig)

# Footer
st.markdown("This dashboard provides insights into the Twitter election data for the 2020 U.S. Presidential election.")