import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS
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
engagement_data = twitter_df.groupby('candidate').agg({'likes': 'sum', 'retweet_count': 'sum'}).reset_index()
fig = px.bar(engagement_data, x='candidate', y=['likes', 'retweet_count'], barmode='group', 
             title="Total Likes and Retweets by Candidate")
st.plotly_chart(fig)

# Visualization 3: Proportion of Likes and Retweets
# Group by candidate and calculate total likes, retweet counts, and tweet counts
engagement_data = twitter_df.groupby('candidate').agg({
    'likes': 'sum',          # Sum of likes for each candidate
    'retweet_count': 'sum',  # Sum of retweets for each candidate
    'tweet_id': 'count'      # Number of tweets (rows) for each candidate
}).reset_index()

# Calculate the proportion of likes and retweets against the total tweets (rows) for each candidate
engagement_data['likes_proportion'] = (engagement_data['likes'] / engagement_data['tweet_id']).round(2) * 100
engagement_data['retweets_proportion'] = (engagement_data['retweet_count'] / engagement_data['tweet_id']).round(2) * 100

# Melt the DataFrame to reshape it for plotting
engagement_data_melted = engagement_data.melt(id_vars='candidate', 
                                              value_vars=['likes_proportion', 'retweets_proportion'],
                                              var_name='Engagement Type', value_name='Proportion')

# Plotting the combined proportions using Plotly
fig = px.bar(engagement_data_melted, x='candidate', y='Proportion', color='Engagement Type',
             barmode='group', title='Proportion of Total Likes and Retweets Against Total Tweets per Candidate (%)',
             labels={'Proportion': 'Proportion', 'candidate': 'Candidate'},
             text=None)

# Display the bar chart in Streamlit
st.plotly_chart(fig)

# Visualization 4: Tweets Over Time
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

filtered_df['country'] = filtered_df['country'].str.strip().str.lower()

# Get the top 5 countries by tweet count
top_countries_tweet = filtered_df['country'].value_counts().nlargest(5)

# Plot the results
fig, ax = plt.subplots()
sns.barplot(x=top_countries_tweet.values, y=top_countries_tweet.index, ax=ax, palette='Set2')
plt.xlabel("Number of Tweets")
plt.title("Top 5 Countries by Tweets")
st.pyplot(fig)

# Get the top 5 countries by tweet count for each candidate
top_countries_by_candidate = (
    filtered_df.groupby(['candidate', 'country'])
    .size()
    .reset_index(name='tweet_count')
)

# Get the top 5 countries for each candidate
top_countries = top_countries_by_candidate.sort_values(['candidate', 'tweet_count'], ascending=[True, False])
top_5_countries_per_candidate = top_countries.groupby('candidate').head(5)

# Plot the results
plt.figure(figsize=(10, 6))
sns.barplot(data=top_5_countries_per_candidate, x='tweet_count', y='country', hue='candidate', palette='Set2')
plt.xlabel("Number of Tweets")
plt.title("Top 5 Countries by Tweets for Each Candidate")
plt.legend(title='Candidate')
plt.tight_layout()

# Show the plot in Streamlit
st.pyplot(plt)

# -------------------- Top US Cities by Tweet Count --------------------
# Filter for tweets from U.S. cities
us_tweets = twitter_df[twitter_df['country'].str.strip().str.lower() == 'united states']

#droppig uknown values 
us_tweets = us_tweets[us_tweets['state'].str.strip().str.lower()!='unknown']

# Count tweets by city
tweets_count_by_state = us_tweets.groupby('state').size().reset_index(name='Tweet Count')

# Get the top 5 cities with the maximum tweets
top_5_states = tweets_count_by_state.nlargest(5, 'Tweet Count')

# creating a bar chart
fig = px.bar(top_5_states,
                x='state',
                y='Tweet Count',
                title='Top 5 US States by Number of Tweets',
                labels={'Tweet Count': 'Number of Tweets', 'states': 'States'},
                color='Tweet Count',
                color_continuous_scale=px.colors.sequential.Viridis)

st.plotly_chart(fig)


# Most Used Devices (Source)
st.header("Most Used Devices (Source)")
top_sources = twitter_df['source'].value_counts().nlargest(5)
fig, ax = plt.subplots()
sns.barplot(x=top_sources.values, y=top_sources.index, ax=ax, palette='Set1')
plt.xlabel("Number of Tweets")
plt.title("Top 5 Devices Used for Tweeting")
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

# Most Engaged User Details
st.header("Most Engaged User Details")
# Group by user_id and sum the likes
likes_by_user = twitter_df.groupby('user_id')['likes'].sum().reset_index()
# Get the user_id with the highest sum of likes
highest_total_likes_user = likes_by_user.loc[likes_by_user['likes'].idxmax()]
# Extracting the user_id and the total likes
user_id_with_highest_total_likes = highest_total_likes_user['user_id']
total_likes = highest_total_likes_user['likes']
# Filter all tweets from the most engaged id_user
most_engaged_user_df = twitter_df.loc[twitter_df["user_id"]==user_id_with_highest_total_likes]
st.write(f"**User ID:** {user_id_with_highest_total_likes}")
st.write(f"**Country:** {most_engaged_user_df['country'].mode().to_list()[0]}")
st.write(f"**Followers:** {most_engaged_user_df['user_followers_count'].max()}")
st.write(f"**Total Likes:** {total_likes}")
st.write(f"**Max Likes in a single tweet:** {most_engaged_user_df['likes'].max()}")
st.write(f"**Tweet Count:** {most_engaged_user_df['user_id'].count()}")

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

# Sentiment Analysis
# Group by candidate and sentiment to get the counts
st.header("Sentiment Analysis")
sentiment_by_candidate = twitter_df.groupby(['candidate', 'sentiment']).size().reset_index(name='count')

# Create the interactive bar chart with specified colors for each sentiment
fig = px.bar(
    sentiment_by_candidate,
    x='candidate',
    y='count',
    color='sentiment',
    title='Sentiment Analysis of Tweets for Biden and Trump',
    color_discrete_map={'positive': '#4caf50', 'neutral': '#ffeb3b', 'negative': '#f44336'},
    barmode='group',
    labels={'count': 'Number of Tweets', 'candidate': 'Candidate', 'sentiment': 'Sentiment'}
)

# Display the interactive Plotly chart in Streamlit
st.plotly_chart(fig)

# Sentiment Proportions
# Group by candidate and sentiment to get the proportional counts
proportion_sentiment_by_candidate = (
    twitter_df.groupby(['candidate', 'sentiment']).size() / twitter_df.groupby('candidate').size()
).reset_index(name='proportion').round(2)

# Create the unstacked bar chart with proportions and specified colors for each sentiment
fig = px.bar(
    proportion_sentiment_by_candidate,
    x='candidate',
    y='proportion',
    color='sentiment',
    title='Proportional Sentiment Analysis of Tweets for Biden and Trump',
    color_discrete_map={'positive': '#4caf50', 'neutral': '#ffeb3b', 'negative': '#f44336'},
    barmode='group'
)

# Display the interactive Plotly chart in Streamlit
st.plotly_chart(fig)


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
st.subheader("WordCloud Analysis")
@st.cache_data
def generate_wordcloud(df, sentiment):
    # Combine all tweet texts for the specific sentiment
    text = ' '.join(df[df['sentiment'] == sentiment]['tweet_cleaned'])
    
    # Add 'amp' to the existing set of stopwords
    stopwords = STOPWORDS.union({'amp'})

    # Generate the word cloud
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white', 
        stopwords=stopwords
    ).generate(text)
    
    # Plot the word cloud
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

# Display word clouds for Biden's positive and negative tweets
st.write("Biden Positive Word Cloud")
fig_biden_positive = generate_wordcloud(biden_df, 'positive')
st.pyplot(fig_biden_positive)

st.write("Biden Negative Word Cloud")
fig_biden_negative = generate_wordcloud(biden_df, 'negative')
st.pyplot(fig_biden_negative)

# Display word clouds for Trump's positive and negative tweets
st.write("Trump Positive Word Cloud")
fig_trump_positive = generate_wordcloud(trump_df, 'positive')
st.pyplot(fig_trump_positive)

st.write("Trump Negative Word Cloud")
fig_trump_negative = generate_wordcloud(trump_df, 'negative')
st.pyplot(fig_trump_negative)


# Display the emoji counts
st.subheader("Top 10 Emojis by candidate")

# Extract all emojis from tweet column and append to a list
biden_emojis_list = []

for tweet in biden_df['tweet']:
    for char in tweet:
        if char in emoji.EMOJI_DATA:  # Check if the character is an emoji
            biden_emojis_list.append(char)

# Count the frequency of emojis
biden_emoji_counts = Counter(biden_emojis_list)

# Extract only the emojis from the top 10 list
biden_top_emojis = [emoji for emoji, _ in biden_emoji_counts.most_common(10)]
# Extract only the frequency from the top 10 list
biden_top_freqs = [freq for _, freq in biden_emoji_counts.most_common(10)]


# Create a bar chart using Plotly
fig = go.Figure(data=[go.Bar(x=biden_top_emojis, y=biden_top_freqs)])
fig.update_layout(
    title='Top 10 Biden Emojis in Tweets',
    xaxis_title='Emojis',
    yaxis_title='Frequency',
    xaxis_tickmode='array'
)

# Show the plot in Streamlit
st.plotly_chart(fig)


# Extract all emojis from tweet column and append to a list
trump_emojis_list = []

for tweet in trump_df['tweet']:
    for char in tweet:
        if char in emoji.EMOJI_DATA:  # Check if the character is an emoji
            trump_emojis_list.append(char)

# Count the frequency of emojis
trump_emoji_counts = Counter(trump_emojis_list)

# Extract only the emojis from the top 10 list
trump_top_emojis = [emoji for emoji, _ in trump_emoji_counts.most_common(10)]
# Extract only the frequency from the top 10 list
trump_top_freqs = [freq for _, freq in trump_emoji_counts.most_common(10)]

# Create a bar chart using Plotly
fig = go.Figure(data=[go.Bar(x=trump_top_emojis, y=trump_top_freqs)])
fig.update_layout(
    title='Top 10 Trump Emojis in Tweets',
    xaxis_title='Emojis',
    yaxis_title='Frequency',
    xaxis_tickmode='array'
)

# Show the plot in Streamlit
st.plotly_chart(fig)


# Footer
st.markdown("This dashboard provides insights into the Twitter election data for the 2020 U.S. Presidential election.")
