import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Load dataset
def load_data():
    return pd.read_csv(r"E:\Redi School\Data Circle 2024\Project\DataCircle_Twitter_Project\twitter_cleaned_data.csv", lineterminator='\n')

# Load data
df_combined = load_data()

# Clean column names (removing '\r')
df_combined.columns = df_combined.columns.str.replace('\r', '')

# Title of the dashboard
st.title("Twitter Election Data Dashboard")

# Display basic information about the dataset
if st.checkbox("Show Data Overview"):
    st.write(df_combined.head())
    st.write("Columns in the dataset:", df_combined.columns.tolist())
    st.write("DataFrame Info:")
    buffer = io.StringIO()
    df_combined.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

# Visualization 1: Total Tweets by Candidate
st.header("Total Tweets by Candidate")
total_tweets = df_combined['candidate'].value_counts()
st.bar_chart(total_tweets)

# Visualization 2: Tweets Over Time
st.header("Tweets Over Time")
df_combined['created_at'] = pd.to_datetime(df_combined['created_at'])
tweets_over_time = df_combined['created_at'].dt.date.value_counts().sort_index()
fig, ax = plt.subplots()
ax.plot(tweets_over_time.index, tweets_over_time.values)
plt.xticks(rotation=45)
plt.xlabel("Date")
plt.ylabel("Number of Tweets")
plt.title("Tweets Over Time")
st.pyplot(fig)

# Visualization 3: Top 5 Countries by Tweets
st.header("Top 5 Countries by Tweets")
top_countries = df_combined['country'].value_counts().nlargest(5)
fig, ax = plt.subplots()
sns.barplot(x=top_countries.values, y=top_countries.index, ax=ax, palette='Set2')
plt.xlabel("Number of Tweets")
plt.title("Top 5 Countries by Tweets")
st.pyplot(fig)

# Visualization 4: Tweet Engagement (Likes and Retweets)
st.header("Tweet Engagement (Likes and Retweets)")
engagement_data = df_combined.groupby('candidate').agg({'likes': 'sum', 'retweet_count': 'sum'}).reset_index()
fig = px.bar(engagement_data, x='candidate', y=['likes', 'retweet_count'], barmode='group', 
             title="Total Likes and Retweets by Candidate")
st.plotly_chart(fig)

# New Visualization 1: Most Active User Details
st.header("Most Active User Details")
most_active_user = df_combined['user_id'].value_counts().idxmax()
user_info = df_combined[df_combined['user_id'] == most_active_user].iloc[0]
st.write(f"**User Name:** {user_info['user_id']}")
st.write(f"**Country:** {user_info['country']}")
st.write(f"**Followers:** {user_info['user_followers_count']}")
st.write(f"**Likes:** {user_info['likes']}")
st.write(f"**Candidate Tweeted:** {user_info['candidate']}")

# New Visualization 2: Most Used Devices (Source)
st.header("Most Used Devices (Source)")
top_sources = df_combined['source'].value_counts().nlargest(5)
fig, ax = plt.subplots()
sns.barplot(x=top_sources.values, y=top_sources.index, ax=ax, palette='Set1')
plt.xlabel("Number of Tweets")
plt.title("Top 5 Devices Used for Tweeting")
st.pyplot(fig)

# Footer
st.markdown("This dashboard provides insights into the Twitter election data for the 2020 U.S. Presidential election.")