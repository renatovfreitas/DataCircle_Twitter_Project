import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from textblob import TextBlob
from wordcloud import WordCloud
import io
import emoji
from collections import Counter
import plotly.graph_objs as go

# Set Streamlit page configuration
st.set_page_config(page_title="Election 2020 Sentiment Analysis", layout="centered")

# Title of the dashboard
st.title("US 2020 Election Sentiment Analysis")

# Load dataset with caching to optimize loading time
@st.cache_data
def load_data():
    return pd.read_csv(r"C:\Users\User\iCloudDrive\Cursos\Data Circle\DataCircle_Twitter_Project\twitter_cleaned_data.csv", lineterminator='\n')

# Load data
twitter_df = load_data()

# Clean column names (removing '\r')
twitter_df.columns = twitter_df.columns.str.replace('\r', '')

# Display basic information about the dataset
if st.checkbox("Show Data Overview"):
    st.write(twitter_df.head())
    st.write("Columns in the dataset:", twitter_df.columns.tolist())
    st.write("DataFrame Info:")
    buffer = io.StringIO()
    twitter_df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

# Visualization 1: Total Tweets by Candidate
st.header("Total Tweets by Candidate")
total_tweets = twitter_df['candidate'].value_counts()
st.bar_chart(total_tweets)

# Visualization 2: Tweet Engagement (Likes and Retweets)
st.header("Tweet Engagement (Likes and Retweets)")

# Group by candidate and sum likes and retweet_count
engagement_data = twitter_df.groupby('candidate').agg({'likes': 'sum', 'retweet_count': 'sum'}).reset_index()

# Calculate total engagement and proportions
engagement_data['total_engagement'] = engagement_data['likes'] + engagement_data['retweet_count']
engagement_data['likes_proportion'] = engagement_data['likes'] / engagement_data['total_engagement']
engagement_data['retweets_proportion'] = engagement_data['retweet_count'] / engagement_data['total_engagement']

# Prepare data for plotting with proportions
engagement_data_melted = engagement_data.melt(id_vars='candidate', value_vars=['likes_proportion', 'retweets_proportion'],
                                              var_name='Engagement Type', value_name='Proportion')

# Plot the proportions using plotly
fig = px.bar(engagement_data_melted, x='candidate', y='Proportion', color='Engagement Type', 
             title="Proportion of Likes and Retweets by Candidate", barmode='group',
             labels={'likes_proportion': 'Likes', 'retweets_proportion': 'Retweets'},
             color_discrete_map={'likes_proportion': '#1f77b4', 'retweets_proportion': '#ff7f0e'})

st.plotly_chart(fig)

# Visualization 3: Tweets Over Time
st.header("Tweets Over Time")
twitter_df['created_at'] = pd.to_datetime(twitter_df['created_at'])
tweets_over_time = twitter_df['created_at'].dt.date.value_counts().sort_index()
fig, ax = plt.subplots()
ax.plot(tweets_over_time.index, tweets_over_time.values)
plt.xticks(rotation=45)
plt.xlabel("Date")
plt.ylabel("Number of Tweets")
plt.title("Tweets Over Time")
st.pyplot(fig)

# Visualization 4: Top 5 Countries by Tweets
st.header("Top 5 Countries by Tweets")

# Remove 'unknown' countries, stripping whitespace and making the comparison case-insensitive
filtered_df = twitter_df[twitter_df['country'].str.strip().str.lower() != 'unknown']

# Get the top 5 countries by tweet count
top_countries = filtered_df['country'].value_counts().nlargest(5)

# Plot the results
fig, ax = plt.subplots()
sns.barplot(x=top_countries.values, y=top_countries.index, ax=ax, palette='Set2')
plt.xlabel("Number of Tweets")
plt.title("Top 5 Countries by Tweets")
st.pyplot(fig)


# Most Active User Details
st.header("Most Active User Details")
most_active_user = twitter_df['user_id'].value_counts().idxmax()
user_info = twitter_df[twitter_df['user_id'] == most_active_user].iloc[0]
st.write(f"**User ID:** {user_info['user_id']}")
st.write(f"**Country:** {user_info['country']}")
st.write(f"**Followers:** {user_info['user_followers_count']}")
st.write(f"**Total Likes:** {twitter_df[twitter_df['user_id']==most_active_user]['likes'].sum()}")
st.write(f"**Max Likes:** {twitter_df[twitter_df['user_id']==most_active_user]['likes'].max()}")
st.write(f"**Tweet Count:** {twitter_df['user_id'].value_counts().max()}")

# Most Used Devices (Source)
st.header("Most Used Devices (Source)")
top_sources = twitter_df['source'].value_counts().nlargest(5)
fig, ax = plt.subplots()
sns.barplot(x=top_sources.values, y=top_sources.index, ax=ax, palette='Set1')
plt.xlabel("Number of Tweets")
plt.title("Top 5 Devices Used for Tweeting")
st.pyplot(fig)

# Sentiment Analysis Function
@st.cache_data
def perform_sentiment_analysis(df):
    def get_sentiment(text):
        analysis = TextBlob(text)
        if analysis.sentiment.polarity > 0:
            return 'positive'
        elif analysis.sentiment.polarity == 0:
            return 'neutral'
        else:
            return 'negative'
    df['sentiment'] = df['tweet_cleaned'].apply(get_sentiment)
    return df

# Apply sentiment analysis
with st.spinner('Analyzing sentiment...'):
    twitter_df = perform_sentiment_analysis(twitter_df)
st.success('Sentiment analysis complete!')

# Sentiment Proportions
st.subheader("Sentiment Proportions")
proportion_sentiment_by_candidate = twitter_df.groupby('candidate')["sentiment"].value_counts(normalize=True).unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(10, 7))
proportion_sentiment_by_candidate.plot(kind='bar', stacked=True, ax=ax, color=['#f44336', '#ffeb3b', '#4caf50'])
plt.title('Sentiment Analysis of Tweets')
plt.xlabel('Sentiment')
plt.ylabel('Proportion of Tweets')
plt.xticks(rotation=0)
st.pyplot(fig)

# Sentiment Trends Over Time
st.subheader("Sentiment Trends Over Time")
twitter_df["created_at_date"] = twitter_df["created_at"].dt.date


biden_df = twitter_df[twitter_df['candidate']=='biden']
trump_df = twitter_df[twitter_df['candidate']=='trump']

# Group by date and sentiment for both datasets
biden_sentiment_over_time = biden_df.groupby(['created_at_date', 'sentiment']).size().unstack(fill_value=0)
trump_sentiment_over_time = trump_df.groupby(['created_at_date', 'sentiment']).size().unstack(fill_value=0)

# Create a figure with 2 subplots (1 row, 2 columns)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7), sharey=True)  # sharey=True ensures both plots have the same y-axis scale

# Plot Biden sentiment trends
biden_sentiment_over_time.plot(kind='line', ax=ax1, color=['#f44336', '#ffeb3b', '#4caf50'])
ax1.set_title('Biden Sentiment Trends Over Time')
ax1.set_xlabel('Date')
ax1.set_ylabel('Number of Tweets')

# Plot Trump sentiment trends
trump_sentiment_over_time.plot(kind='line', ax=ax2, color=['#f44336', '#ffeb3b', '#4caf50'])
ax2.set_title('Trump Sentiment Trends Over Time')
ax2.set_xlabel('Date')

# Add overall title and adjust layout
fig.suptitle('Sentiment Trends Over Time (Biden vs Trump)')
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Render the figure in Streamlit
st.pyplot(fig)

# Word Clouds
# Function to generate word cloud for a particular sentiment
@st.cache_data
def generate_wordcloud(df, sentiment):
    text = ' '.join(df[df['sentiment'] == sentiment]['tweet_cleaned'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

# Word clouds for Biden's positive, negative, and neutral tweets
st.write("Biden Positive Word Cloud")
generate_wordcloud(biden_df, 'positive')
st.write("Biden Negative Word Cloud")
generate_wordcloud(biden_df, 'negative')

# Word clouds for Trump's positive, negative, and neutral tweets
st.write("Trump Positive Word Cloud")
generate_wordcloud(trump_df, 'positive')
st.write("Trump Negative Word Cloud")
generate_wordcloud(trump_df, 'negative')


# Extract all emojis from tweet column and append to a list
all_emojis_list = []

for tweet in twitter_df['tweet']:
    for char in tweet:
        if char in emoji.EMOJI_DATA:  # Check if the character is an emoji
            all_emojis_list.append(char)

# Count the frequency of emojis
emoji_counts = Counter(all_emojis_list)

# Extract only the emojis from the top 10 list
top_emojis = [emoji for emoji, _ in emoji_counts.most_common(10)]
# Extract only the frequency from the top 10 list
top_freqs = [freq for _, freq in emoji_counts.most_common(10)]

# Streamlit App
st.title("Emoji Analysis in Tweets")

# Display the emoji counts
st.subheader("Top 10 Emojis and Their Frequencies")

# Create a bar chart using Plotly
fig = go.Figure(data=[go.Bar(x=top_emojis, y=top_freqs)])
fig.update_layout(
    title='Top 10 Emojis in Tweets',
    xaxis_title='Emojis',
    yaxis_title='Frequency',
    xaxis_tickmode='array'
)

# Show the plot in Streamlit
st.plotly_chart(fig)

# Footer
st.markdown("This dashboard provides insights into the Twitter election data for the 2020 U.S. Presidential election.")
