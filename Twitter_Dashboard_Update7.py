# Required libraries
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
import plotly.graph_objs as go
import emoji
import io
from datetime import datetime  # Ensure this import is included

# Load dataset with caching
@st.cache_data
def load_data():
    return pd.read_csv(r"E:\Redi School\Data Circle 2024\Project\DataCircle_Twitter_Project\twitter_cleaned_data_update.csv", lineterminator='\n')

# Load data
twitter_df = load_data()

# Clean column names
twitter_df.columns = twitter_df.columns.str.replace('\r', '')

# Sidebar for candidate selection
st.sidebar.title("Select Candidate")
candidate = st.sidebar.selectbox("Choose a candidate:", ["Biden", "Trump"])

# Filter DataFrame based on selected candidate
filtered_df = twitter_df[twitter_df['candidate'].str.lower() == candidate.lower()].copy()  # Use .copy()

# Sentiment analysis function
def get_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity == 0:
        return 'neutral'
    else:
        return 'negative'

# Apply sentiment analysis to filtered DataFrame
filtered_df['sentiment'] = filtered_df['tweet_cleaned'].apply(get_sentiment)  # No need for .loc here

# Convert created_at to datetime
filtered_df['created_at'] = pd.to_datetime(filtered_df['created_at'], errors='coerce')  # Ensure conversion

# Display candidate images
if candidate == "Biden":
    st.image("E:\Redi School\Data Circle 2024\Project\DataCircle_Twitter_Project\joe_biden_president_cartoon_portrait_6919399.jpg", caption="Joe Biden", width=300)
else:
    st.image("E:\Redi School\Data Circle 2024\Project\DataCircle_Twitter_Project\donald_trump_PNG40.png", caption="Donald Trump", width=300)

# Visualization 1: Total Tweets by Candidate
tab1, tab2, tab3, tab4 = st.tabs(["Total Tweets for Candidate", "Tweet Engagement", "Sentiment Analysis", "Word Clouds"])

with tab1:
    st.header("Total Tweets per Day by Candidate")
    
    # Group by date and candidate to get the count of tweets per day
    daily_tweets = filtered_df.groupby(filtered_df['created_at'].dt.date).size()

    # Plotting
    daily_tweets.plot(kind='bar', figsize=(10, 5), color='blue' if candidate == 'Biden' else 'red')
    plt.title(f"Daily Tweets Count for {candidate}")
    plt.xlabel("Date")
    plt.ylabel("Number of Tweets")
    plt.xticks(rotation=45)
    
    # Show the plot in Streamlit
    st.pyplot(plt)
    plt.clf()  # Clear the figure after showing

# Visualization 2: Tweet Engagement (Likes and Retweets)
with tab2:
    st.header("Tweet Engagement (Likes and Retweets)")
    engagement_data = filtered_df.groupby('candidate').agg({'likes': 'sum', 'retweet_count': 'sum'}).reset_index()
    fig = px.bar(engagement_data, x='candidate', y=['likes', 'retweet_count'], barmode='group', 
                 title="Total Likes and Retweets by Candidate", color_discrete_sequence=['blue' if candidate == 'Biden' else 'red'])
    st.plotly_chart(fig)

# Visualization 3: Sentiment Analysis
with tab3:
    st.header("Sentiment Analysis")

    # Group by sentiment to get the counts
    sentiment_counts = filtered_df['sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']

    # Create pie chart for sentiment proportions
    fig = px.pie(sentiment_counts, values='Count', names='Sentiment', title='Sentiment Distribution', 
                 color_discrete_sequence=['#4caf50', '#ffeb3b', '#f44336'])
    st.plotly_chart(fig)

    # Sentiment Trends Over Time st.header("Sentiment Trends Over Time")

    # Group by date and sentiment for Biden
    sentiment_trends = filtered_df.groupby([pd.Grouper(key='created_at', freq='D'), 'sentiment']).size().unstack(fill_value=0)

    # Plot sentiment trends
    fig = go.Figure()
    for sentiment in sentiment_trends.columns:
        fig.add_trace(go.Scatter(
            x=sentiment_trends.index,
            y=sentiment_trends[sentiment],
            mode='lines',
            name=sentiment
        ))

    # Add vertical line for election day
    election_day = datetime(2020, 11, 3)
    fig.add_vline(x=election_day, line_width=2, line_dash="dash", line_color="red")

    # Add annotation for election day
    fig.add_annotation(
        x=election_day,
        y=max(sentiment_trends.max()),
        text="Election Day",
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-40,
        font=dict(color="red")
    )

    fig.update_layout(title=f'Sentiment Trends Over Time for {candidate}', xaxis_title='Date', yaxis_title='Number of Tweets')
    st.plotly_chart(fig)

# Visualization 4: Word Clouds
with tab4:
    st.header("WordCloud Analysis")

    def generate_wordcloud(df, sentiment):
        text = ' '.join(df[df['sentiment'] == sentiment]['tweet_cleaned'])
        stopwords = STOPWORDS.union({'amp'})
        wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords).generate(text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        return fig

    st.write("Positive Word Cloud")
    fig_positive = generate_wordcloud(filtered_df, 'positive')
    st.pyplot(fig_positive)

    st.write("Negative Word Cloud")
    fig_negative = generate_wordcloud(filtered_df, 'negative')
    st.pyplot(fig_negative)

# Emoji Analysis
with tab4:
    st.header("Emoji Analysis")
    all_emojis_list = []
    recognized_emojis = set(emoji.EMOJI_DATA.keys())

    for tweet in twitter_df['tweet']:
        for char in tweet:
            if char in recognized_emojis:
                all_emojis_list.append(char)

    # Frequency count of emojis
    emoji_counts = Counter(all_emojis_list)
    top_emojis = emoji_counts.most_common(10)

    # Check if there are emojis to create the bar chart
    if top_emojis:
        # Create DataFrame for the top emojis
        emoji_df = pd.DataFrame(top_emojis, columns=['Emoji', 'Frequency'])

        # Plot bar chart
        fig = go.Figure(data=[go.Bar(x=emoji_df['Emoji'], y=emoji_df['Frequency'], marker_color='blue')])
        fig.update_layout(title='Top Emojis Used in Tweets', xaxis_title='Emoji', yaxis_title='Frequency')
        st.plotly_chart(fig)

# Footer
st.markdown("This dashboard provides insights into the Twitter election data for the 2020 U.S. Presidential election.")