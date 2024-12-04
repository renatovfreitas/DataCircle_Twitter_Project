import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px

# Load dataset with caching
@st.cache_data
def load_data():
    return pd.read_csv(r"/Users/may/Documents/GitHub/DataCircle_Twitter_Project/twitter_cleaned_data.csv", lineterminator='\n')

# Load the data
twitter_df = load_data()

# Ensure the 'created_at' column is in datetime format
twitter_df['created_at'] = pd.to_datetime(twitter_df['created_at'], errors='coerce')

# First Graph: COVID Mentions Over Time
# Check if 'tweet_cleaned' column exists
if 'tweet_cleaned' in twitter_df.columns:
    # Filter for tweets mentioning "COVID" (and variations)
    covid_mentions = twitter_df[twitter_df['tweet_cleaned'].str.contains(r"covid|covid-19|covid2020|covid19", case=False, na=False)]
else:
    st.error("The dataset does not contain a 'tweet_cleaned' column. Please check the column names.")

# Count mentions by date
covid_daily = covid_mentions.groupby(covid_mentions['created_at'].dt.date).size().reset_index(name='mention_count')

# Visualization 1: COVID Mentions Over Time
fig1 = go.Figure()

fig1.add_trace(go.Scatter(
    x=covid_daily['created_at'],
    y=covid_daily['mention_count'],
    mode='lines+markers',
    line=dict(shape='spline'),
    marker=dict(color='rgb(204, 204, 255)', size=6),
    name='COVID Mentions'
))

fig1.update_layout(
    title='Volume of COVID Mentions Over Time',
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

# Calculate the percentage of tweets containing "COVID"
total_tweets = len(twitter_df)
covid_mentions_total = len(covid_mentions)
covid_percentage = (covid_mentions_total / total_tweets) * 100

# Show total number of tweets and COVID mentions
st.markdown(f"<h3 style='color:white;'>Total tweets: {total_tweets:,}</h3>", unsafe_allow_html=True)
st.markdown(f"<h3 style='color:white;'>Total tweets mentioning COVID: {covid_mentions_total:,} ({covid_percentage:.2f}% of total tweets)</h3>", unsafe_allow_html=True)

# Second Graph: COVID mentions by State (Top 3 States) without Unknown States
if 'state' in twitter_df.columns:
    # Clean the dataset by removing 'Unknown' (case-insensitive) and non-state entries
    twitter_df_cleaned = twitter_df[
        (twitter_df['state'].notnull()) & 
        (twitter_df['state'] != '') & 
        (~twitter_df['state'].str.lower().eq('unknown'))  # Exclude 'Unknown' (case-insensitive)
    ]
    
    # Filter for tweets containing COVID-related keywords in the cleaned dataset
    state_covid_mentions = twitter_df_cleaned[
        twitter_df_cleaned['tweet_cleaned'].str.contains(r"covid|covid-19|covid2020|covid19", case=False, na=False)
    ]
    
    # Count COVID mentions by state
    state_covid_count = state_covid_mentions['state'].value_counts().reset_index()
    state_covid_count.columns = ['state', 'covid_mention_count']
else:
    st.error("The dataset does not contain a 'state' column. Please check the column names.")

# Calculate the total mentions
total_covid_mentions = len(covid_mentions)

# Sort the states by COVID mentions and select top 3
top_states = state_covid_count.head(3)

# Calculate the percentage of total tweets mentioning COVID for the top 3 states
top_states['covid_percentage'] = (top_states['covid_mention_count'] / total_covid_mentions) * 100

# Visualization: COVID Mentions by State (Top 3 States)
fig2 = px.bar(
    top_states,
    x='state',
    y='covid_mention_count',
    title='Top 3 U.S. States with the Highest COVID Mentions',
    labels={'covid_mention_count': 'Number of COVID Mentions', 'state': 'State'},
    template='plotly_dark',
    color='state',
    color_discrete_sequence=px.colors.qualitative.Set3
)

# Update layout for better presentation
fig2.update_layout(
    title_font=dict(size=22, color='white', family="Arial, sans-serif"),
    xaxis_title="State",
    yaxis_title="COVID Mentions",
    plot_bgcolor='black',
    paper_bgcolor='black',
    font=dict(color='white'),
    xaxis=dict(tickangle=45),
    showlegend=False
)

# Display the plot in Streamlit
st.plotly_chart(fig2, use_container_width=True)

# Display the percentages for the top 3 states
for _, row in top_states.iterrows():
    st.markdown(f"<h3 style='color:white;'>State: {row['state']} - COVID Mentions: {row['covid_mention_count']:,} ({row['covid_percentage']:.2f}% of total mentions)</h3>", unsafe_allow_html=True)

# Insights
st.markdown("""
    <div style="color:white; font-size:16px;">
        <p>**Insights:**</p>
        <ul>
            <li>The top 3 states have a significant proportion of COVID-related tweets, which may indicate a higher focus on the pandemic in these regions.</li>
            <li>This analysis highlights the regional differences in COVID-related discourse across the U.S.</li>
        </ul>
    </div>
""", unsafe_allow_html=True)
