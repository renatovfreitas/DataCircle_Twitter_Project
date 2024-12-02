import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime

# Load dataset with caching
@st.cache_data
def load_data():
    return pd.read_csv(r"/Users/may/Documents/GitHub/DataCircle_Twitter_Project/twitter_cleaned_data.csv", lineterminator='\n')

# Load the data
twitter_df = load_data()

# Ensure the 'created_at' column is in datetime format
twitter_df['created_at'] = pd.to_datetime(twitter_df['created_at'], errors='coerce')

# Check if 'tweet_cleaned' column exists in the dataset
if 'tweet_cleaned' in twitter_df.columns:
    covid_trump_mentions = twitter_df[
        twitter_df['tweet_cleaned'].str.contains("covid|pandemic|covid-19", case=False, na=False) & 
        twitter_df['tweet_cleaned'].str.contains("trump", case=False, na=False)
    ]
else:
    st.error("The dataset does not contain a 'tweet_cleaned' column. Please check the column names.")

# Count mentions by date
covid_trump_daily = covid_trump_mentions.groupby(covid_trump_mentions['created_at'].dt.date).size().reset_index(name='mention_count')

# Visualization 1: Mentions Over Time (Line chart)
fig1 = go.Figure()

fig1.add_trace(go.Scatter(
    x=covid_trump_daily['created_at'],
    y=covid_trump_daily['mention_count'],
    mode='lines+markers',
    line=dict(shape='spline'),
    marker=dict(color='rgb(204, 204, 255)', size=6),
    name='Mentions of COVID and Trump'
))

fig1.update_layout(
    title='Volume of Mentions Over Time',
    title_font=dict(size=22, color='white', family="Arial, sans-serif"),
    xaxis_title="Date",
    yaxis_title="Number of Mentions",
    plot_bgcolor='black',
    paper_bgcolor='black',
    font=dict(color='white'),
    xaxis=dict(tickangle=45),
    yaxis=dict(tickprefix="Tweets: "),
    showlegend=False
)

# Display the plot in Streamlit
st.plotly_chart(fig1, use_container_width=True)

# Total mentions of tweets mentioning both Trump and COVID
total_mentions = len(covid_trump_mentions)
st.markdown(f"<h3 style='color:white;'>Total tweets mentioning COVID and Trump: {total_mentions:,}</h3>", unsafe_allow_html=True)

st.markdown("""
    <div style="color:white; font-size:16px;">
        <p>**Insights:**</p>
        <ul>
            <li>The pandemic dominated public discourse during the 2020 U.S. Presidential Election.</li>
            <li>Trump’s handling of the pandemic was a major topic of debate, both positively and negatively.</li>
            <li>An increase in mentions could correlate with key events, such as policy announcements, case spikes, or debates.</li>
        </ul>
        <p>**Potential Impact on Campaign:**</p>
        <ul>
            <li>Negative sentiments around COVID-related tweets may have affected voter perceptions.</li>
            <li>Policy-related mentions could highlight areas of public concern or dissatisfaction.</li>
        </ul>
    </div>
""", unsafe_allow_html=True)

# Calculate the average likes over time
covid_trump_daily_likes = covid_trump_mentions.groupby(covid_trump_mentions['created_at'].dt.date)['likes'].mean().reset_index(name='average_likes')

# Visualization 2: Average Likes Over Time
fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    x=covid_trump_daily_likes['created_at'],
    y=covid_trump_daily_likes['average_likes'],
    mode='lines+markers',
    line=dict(shape='spline'),
    marker=dict(color='rgb(255, 204, 0)', size=6),
    name='Average Likes'
))

fig2.update_layout(
    title='Average Likes Over Time for Tweets Mentioning COVID and Trump',
    title_font=dict(size=22, color='white', family="Arial, sans-serif"),
    xaxis_title="Date",
    yaxis_title="Average Likes",
    plot_bgcolor='black',
    paper_bgcolor='black',
    font=dict(color='white'),
    xaxis=dict(tickangle=45),
    yaxis=dict(tickprefix="Likes: "),
    showlegend=False
)

# Display the plot in Streamlit
st.plotly_chart(fig2, use_container_width=True)

# Filter out unwanted states, including "england" with a lowercase "e"
valid_states = covid_trump_mentions[~covid_trump_mentions['state'].isin(['unknown', 'england', 'non-us'])]
valid_states_likes = valid_states.groupby('state')['likes'].mean().reset_index(name='average_likes')

# Get the top 5 states by volume of mentions
state_mention_count = valid_states['state'].value_counts().reset_index()
state_mention_count.columns = ['state', 'mention_count']
top_5_states = state_mention_count.head(5)

# Merge the top 5 states with their average likes
top_5_states_likes = top_5_states.merge(valid_states_likes, on='state', how='left')

# Visualization 3: Modern Bar chart for Average Likes by Top 5 US States
fig3 = px.bar(
    top_5_states_likes,
    x='state',
    y='average_likes',
    color='state',
    title='Average Number of Likes of 5 Top US States',
    labels={'average_likes': 'Average Likes', 'state': 'State'},
    template='plotly_dark',
    color_discrete_sequence=px.colors.qualitative.Set1
)

fig3.update_layout(
    title_font=dict(size=22, color='white', family="Arial, sans-serif"),
    xaxis_title="State",
    yaxis_title="Average Likes",
    plot_bgcolor='black',
    paper_bgcolor='black',
    font=dict(color='white'),
    xaxis=dict(tickangle=45),
    showlegend=False
)

# Display the plot in Streamlit
st.plotly_chart(fig3, use_container_width=True)

