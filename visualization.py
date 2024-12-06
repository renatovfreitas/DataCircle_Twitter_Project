# Required libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates
import plotly.graph_objs as go
from plotly.subplots import make_subplots
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

# Create a column for 'created_at' with date only (without time info)
twitter_df["created_at_date"] = twitter_df["created_at"].dt.date

# Convert to datetime
twitter_df["created_at_date"] = pd.to_datetime(twitter_df["created_at_date"])

# Create a Dataframe for Biden and Trump separately
biden_df = twitter_df[twitter_df['candidate'] == 'biden']
trump_df = twitter_df[twitter_df['candidate'] == 'trump']

# Create tabs
tab1, tab2, tab3 = st.tabs(["Exploratory Data Analysis", "Sentiment Analysis", "WordCloud Analysis"])

# Create variables for KPIs
biden_tweet_count = twitter_df[twitter_df['candidate']=='biden']['candidate'].count()
trump_tweet_count = twitter_df[twitter_df['candidate']=='trump']['candidate'].count()
biden_total_likes = twitter_df[twitter_df['candidate']=='biden']['likes'].sum()
trump_total_likes = twitter_df[twitter_df['candidate']=='trump']['likes'].sum()

# Visualizations in tab 1
with tab1:
    # Create columns for KPIs
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label=r"Biden's total tweets:", value=f"{biden_tweet_count:,}")
    with col2:
        st.metric(label=r"Trump's total tweets:", value=f"{trump_tweet_count:,}")
    col3, col4 = st.columns(2)
    with col3:
        st.metric(label=r"Biden's total likes:", value=f"{biden_total_likes:,}")
    with col4:
        st.metric(label=r"Trump's total likes:", value=f"{trump_total_likes:,}")


    # Visualization 1: Total Tweets by Candidate
    
    # Header
    st.header("Total Tweets per Day by Candidate")

    # Group by date and candidate to get the count of tweets per day
    daily_tweets = (twitter_df.groupby([pd.Grouper(key='created_at', freq='D'), 'candidate']).size().reset_index(name='count'))

    # Rename columns for clarity
    daily_tweets.rename(columns={'created_at': 'date'}, inplace=True)

    # Define custom colors for the candidates
    candidate_colors = {
        'biden': 'blue',
        'trump': 'red'
    }

    # Create the Plotly line chart
    fig = px.line(
        daily_tweets,
        x='date',
        y='count',
        color='candidate',
        markers=True,
        title="Daily Tweets Count by Candidate",
        labels={'date': 'Date', 'count': 'Number of Tweets'},
        color_discrete_map=candidate_colors  # Apply custom colors
    )

    # Capitalize legend labels
    fig.for_each_trace(lambda trace: trace.update(name=trace.name.capitalize()))

    # Customize the layout
    fig.update_layout(
        title_font_size=20,
        xaxis_title="Date",
        yaxis_title="Number of Tweets",
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True, tickformat=','),
        legend_title="Candidate",
        legend=dict(font=dict(size=12)),
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # Visualization 2: Tweet Engagement (Likes and Retweets)

    # Plotting
    st.header("Tweet Engagement (Likes and Retweets)")

    # Group by candidate to get total likes and retweets
    engagement_data = twitter_df.groupby('candidate').agg({'likes': 'sum', 'retweet_count': 'sum'}).reset_index()

    # Capitalize the candidate names for x-axis labels
    engagement_data['candidate'] = engagement_data['candidate'].str.capitalize()

    # Create the plot with Plotly Express
    fig = px.bar(
        engagement_data,
        x='candidate',
        y=['likes', 'retweet_count'],
        barmode='group',
        title="Total Likes and Retweets by Candidate",
        labels={'value': 'Count', 'variable': 'Engagement Type'},
        color_discrete_sequence=['royalblue', 'darkblue']
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

    # Add a dropdown filter for candidate selection
    candidate = st.selectbox("Select a candidate:", options=['Trump', 'Biden'])

    # Filter data for the selected candidate
    filtered_data_geo = tweets_by_state_and_candidate[tweets_by_state_and_candidate['candidate'].str.strip().str.lower() == candidate.lower()]

    # Visualization 3: Geoplot
    st.header("Tweets by States")

    # Create a choropleth map for tweet counts by state for the selected candidate
    fig_geo = px.choropleth(
        filtered_data_geo,
        locations='state',
        locationmode="USA-states",
        color='Tweet Count',
        color_continuous_scale="Viridis",
        scope="usa",
        title=f'Tweet Count by US State for {candidate}',
        labels={'state': 'State', 'Tweet Count': 'Number of Tweets'}
    )

    # Display the map
    st.plotly_chart(fig_geo)

    # Visualization 7: Emoji Analysis
    st.header("Emoji Analysis")

    # Function to load emoji data from CSV and get top 5 emojis with their percentages
    def load_top_emojis(csv_file):
        df = pd.read_csv(csv_file)
        total_frequency = df['Frequency'].sum()  # Calculate total frequency
        df['Percentage'] = (df['Frequency'] / total_frequency) * 100  # Calculate percentage
        top_emojis = df.nlargest(5, 'Frequency')  # Get top 5 emojis
        return top_emojis

    # Load the appropriate emoji data for the selected candidate
    emoji_file = 'biden_emojis.csv' if candidate.lower() == 'biden' else 'trump_emojis.csv'
    emojis = load_top_emojis(emoji_file)

    # Create a bar chart for the selected candidate
    fig_emojis = px.bar(
        emojis, 
        x='Emoji', 
        y='Percentage', 
        title=f'Top 5 Emojis for {candidate} (by Percentage)',
        labels={'Emoji': 'Emoji', 'Percentage': 'Percentage (%)'},
        color_discrete_sequence=['blue'] if candidate.lower() == 'biden' else ['red']
    )

    # Dynamically adjust y-axis range based on data
    fig_emojis.update_yaxes(range=[0, emojis['Percentage'].max() + 5])

    # Display the emoji bar chart
    st.plotly_chart(fig_emojis)

# Visualizations in tab 2
with tab2:
    # Visualization 4: Sentiment Analysis

    # Create pie charts
    st.header("Sentiment Analysis by Candidate")

    # Group by sentiment for each candidate
    biden_sentiment_counts = biden_df['sentiment'].value_counts().reset_index()
    biden_sentiment_counts.columns = ['Sentiment', 'Count']

    trump_sentiment_counts = trump_df['sentiment'].value_counts().reset_index()
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


    # Visualization 5: Sentiment Trends Over Time 

    # Biden's Polarity Sentiment Means Over Time
    biden_sentiment_means = biden_df.groupby('created_at_date')['polarity'].mean()

    # Trump's Polarity Sentiment Means Over Time
    trump_sentiment_means = trump_df.groupby('created_at_date')['polarity'].mean()

    # Ensure datetime index for plotting
    biden_sentiment_means.index = pd.to_datetime(biden_sentiment_means.index)
    trump_sentiment_means.index = pd.to_datetime(trump_sentiment_means.index)

    # Create a figure
    fig = go.Figure()

    # Plot sentiment trends for Biden
    fig.add_trace(go.Scatter(
        x=biden_sentiment_means.index,
        y=biden_sentiment_means.values,
        mode='lines',
        name='Biden',
        line=dict(color='blue')
    ))

    # Plot sentiment trends for Trump
    fig.add_trace(go.Scatter(
        x=trump_sentiment_means.index,
        y=trump_sentiment_means.values,
        mode='lines',
        name='Trump',
        line=dict(color='red')
    ))

    # Update layout
    fig.update_layout(
        title='Sentiment Polarity Means Over Time',
        xaxis=dict(title='Date', tickformat='%Y-%m-%d'),
        yaxis=dict(title='Polarity Mean', range=[0, 0.15]),
        height=500,
        legend=dict(title='Legend'),
        template='plotly_white'
    )

    # Show the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)


    # Visualization 6: Polarity Difference Over Time 
    # Convert trumps daily means to a dataframe
    trump_daily_means_df = pd.DataFrame(trump_sentiment_means).reset_index()

    # Convert Bidens daily means to a dataframe
    biden_daily_means_df = pd.DataFrame(biden_sentiment_means).reset_index()

    # Convert the difference between both candidates daily means to a dataframe
    diff_daily_means_df = pd.DataFrame(
        {
        'Date': trump_daily_means_df['created_at_date'], 
        'Difference': biden_daily_means_df['polarity'] - trump_daily_means_df['polarity']
        }
        )
    
    # Ensure that the 'Date' column in diff_daily_means_df is in datetime format
    diff_daily_means_df['Date'] = pd.to_datetime(diff_daily_means_df['Date'])

    # Plot with Plotly
    fig = go.Figure()

    # Add the main line for sentiment polarity difference
    fig.add_trace(go.Scatter(
        x=diff_daily_means_df['Date'],
        y=diff_daily_means_df['Difference'],
        mode='lines',
        line=dict(color='white'),
        name='Difference in Sentiment Polarity'
    ))

    # Update layout with axis labels, limits, and title
    fig.update_layout(
        title='Sentiment Polarity Means Difference Over Time',
        xaxis_title='Date',
        yaxis_title='Polarity Difference',
        yaxis=dict(range=[0, 0.1], tick0=0, dtick=0.05),
        xaxis=dict(tickformat='%Y-%m-%d'
        ),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )

    # Display the Plotly chart in Streamlit
    st.plotly_chart(fig)

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


    # Whole Period Word Cloud
    st.subheader("Whole Period")

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
    ax1.set_title("Biden's Positive", fontsize=16)

    # Display Trump's word cloud
    ax2.imshow(trump_positive_wordcloud, interpolation='bilinear')
    ax2.axis('off')  # Hide axes
    ax2.set_title("Trump's Positive", fontsize=16)

    # Show the plot in Streamlit
    st.pyplot(positive_fig)

    # Create side-by-side Negative word clouds
    negative_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # Display Biden's word cloud
    ax1.imshow(biden_negative_wordcloud, interpolation='bilinear')
    ax1.axis('off')  # Hide axes
    ax1.set_title("Biden's Negative", fontsize=16)

    # Display Trump's word cloud
    ax2.imshow(trump_negative_wordcloud, interpolation='bilinear')
    ax2.axis('off')  # Hide axes
    ax2.set_title("Trump's Negative", fontsize=16)

    # Show the plot in Streamlit
    st.pyplot(negative_fig)

    # Generate word clouds for both candidates
    biden_negative_hashtag_wordcloud = generate_wordcloud('biden_hashtag_negative_wordcloud.csv')
    trump_negative_hashtag_wordcloud = generate_wordcloud('trump_hashtag_negative_wordcloud.csv')

    # Create side-by-side Negative word clouds
    negative_hashtag_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # Display Biden's word cloud
    ax1.imshow(biden_negative_hashtag_wordcloud, interpolation='bilinear')
    ax1.axis('off')  # Hide axes
    ax1.set_title("Biden's Negative Hashtag", fontsize=16)

    # Display Trump's word cloud
    ax2.imshow(trump_negative_hashtag_wordcloud, interpolation='bilinear')
    ax2.axis('off')  # Hide axes
    ax2.set_title("Trump's Negative Hashtag", fontsize=16)

    # Show the plot in Streamlit
    st.pyplot(negative_hashtag_fig)


    # 16/10/20 Word Cloud
    st.subheader("Town Hall - 16/10/20")
    # Generate word clouds for both candidates
    biden_16_10_positive_wordcloud = generate_wordcloud('biden_16_10_positive_wordcloud.csv')
    biden_16_10_negative_wordcloud = generate_wordcloud('biden_16_10_negative_wordcloud.csv')
    trump_16_10_positive_wordcloud = generate_wordcloud('trump_16_10_positive_wordcloud.csv')
    trump_16_10_negative_wordcloud = generate_wordcloud('trump_16_10_negative_wordcloud.csv')
    
    # Create side-by-side Positive word clouds
    positive_fig_16_10, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Display Biden's word cloud
    ax1.imshow(biden_16_10_positive_wordcloud, interpolation='bilinear')
    ax1.axis('off')  # Hide axes
    ax1.set_title("Biden's Positive", fontsize=16)

    # Display Trump's word cloud
    ax2.imshow(trump_16_10_positive_wordcloud, interpolation='bilinear')
    ax2.axis('off')  # Hide axes
    ax2.set_title("Trump's Positive", fontsize=16)

    # Show the plot in Streamlit
    st.pyplot(positive_fig_16_10)

    # Create side-by-side Negative word clouds
    negative_fig_16_10, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # Display Biden's word cloud
    ax1.imshow(biden_16_10_negative_wordcloud, interpolation='bilinear')
    ax1.axis('off')  # Hide axes
    ax1.set_title("Biden's Negative", fontsize=16)

    # Display Trump's word cloud
    ax2.imshow(trump_16_10_negative_wordcloud, interpolation='bilinear')
    ax2.axis('off')  # Hide axes
    ax2.set_title("Trump's Negative", fontsize=16)

    # Show the plot in Streamlit
    st.pyplot(negative_fig_16_10)


    # 16/10/20 Hashtag Word Cloud
    # Generate word clouds for both candidates
    biden_16_10_hashtag_negative_wordcloud = generate_wordcloud('biden_16_10_hashtag_negative_wordcloud.csv')
    trump_16_10_hashtag_negative_wordcloud = generate_wordcloud('trump_16_10_hashtag_negative_wordcloud.csv')
    
    # Create side-by-side Negative word clouds
    negative_hashtag_16_10_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # Display Biden's word cloud
    ax1.imshow(biden_16_10_hashtag_negative_wordcloud, interpolation='bilinear')
    ax1.axis('off')  # Hide axes
    ax1.set_title("Biden's Negative Hashtag", fontsize=16)

    # Display Trump's word cloud
    ax2.imshow(trump_16_10_hashtag_negative_wordcloud, interpolation='bilinear')
    ax2.axis('off')  # Hide axes
    ax2.set_title("Trump's Negative Hashtag", fontsize=16)

    # Show the plot in Streamlit
    st.pyplot(negative_hashtag_16_10_fig)
    

    # 23/10/20 Word Cloud
    st.subheader("Last Debate - 23/10/20")
    
    # Generate filtered word clouds for both candidates
    biden_23_10_positive_wordcloud = generate_wordcloud('biden_23_10_positive_wordcloud.csv')
    biden_23_10_negative_wordcloud = generate_wordcloud('biden_23_10_negative_wordcloud.csv')
    trump_23_10_positive_wordcloud = generate_wordcloud('trump_23_10_positive_wordcloud.csv')
    trump_23_10_negative_wordcloud = generate_wordcloud('trump_23_10_negative_wordcloud.csv')

    # Create side-by-side Positive word clouds
    positive_23_10_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # Display Biden's word cloud
    ax1.imshow(biden_23_10_positive_wordcloud, interpolation='bilinear')
    ax1.axis('off')  # Hide axes
    ax1.set_title("Biden's Positive", fontsize=16)

    # Display Trump's word cloud
    ax2.imshow(trump_23_10_positive_wordcloud, interpolation='bilinear')
    ax2.axis('off')  # Hide axes
    ax2.set_title("Trump's Positive", fontsize=16)

    # Show the plot in Streamlit
    st.pyplot(positive_23_10_fig)

    # Create side-by-side Negative word clouds
    negative_23_10_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # Display Biden's word cloud
    ax1.imshow(biden_23_10_negative_wordcloud, interpolation='bilinear')
    ax1.axis('off')  # Hide axes
    ax1.set_title("Biden's Negative", fontsize=16)

    # Display Trump's word cloud
    ax2.imshow(trump_23_10_negative_wordcloud, interpolation='bilinear')
    ax2.axis('off')  # Hide axes
    ax2.set_title("Trump's Negative", fontsize=16)

    # Show the plot in Streamlit
    st.pyplot(negative_23_10_fig)


    # 23/10/20 Hashtag Word Cloud
    # Generate word clouds for both candidates
    biden_23_10_hashtag_negative_wordcloud = generate_wordcloud('biden_23_10_hashtag_negative_wordcloud.csv')
    trump_23_10_hashtag_negative_wordcloud = generate_wordcloud('trump_23_10_hashtag_negative_wordcloud.csv')
    
    # Create side-by-side Negative word clouds
    negative_hashtag_23_10_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # Display Biden's word cloud
    ax1.imshow(biden_23_10_hashtag_negative_wordcloud, interpolation='bilinear')
    ax1.axis('off')  # Hide axes
    ax1.set_title("Biden's Negative Hashtag", fontsize=16)

    # Display Trump's word cloud
    ax2.imshow(trump_23_10_hashtag_negative_wordcloud, interpolation='bilinear')
    ax2.axis('off')  # Hide axes
    ax2.set_title("Trump's Negative Hashtag", fontsize=16)

    # Show the plot in Streamlit
    st.pyplot(negative_hashtag_23_10_fig)


    # 03/11/20 Word Cloud
    st.subheader("Election Day - 03/11/20")
    # Generate filtered word clouds for both candidates
    biden_03_11_positive_wordcloud = generate_wordcloud('biden_03_11_positive_wordcloud.csv')
    biden_03_11_negative_wordcloud = generate_wordcloud('biden_03_11_negative_wordcloud.csv')
    trump_03_11_positive_wordcloud = generate_wordcloud('trump_03_11_positive_wordcloud.csv')
    trump_03_11_negative_wordcloud = generate_wordcloud('trump_03_11_negative_wordcloud.csv')

    # Create side-by-side Positive word clouds
    positive_03_11_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # Display Biden's word cloud
    ax1.imshow(biden_03_11_positive_wordcloud, interpolation='bilinear')
    ax1.axis('off')  # Hide axes
    ax1.set_title("Biden's Positive", fontsize=16)

    # Display Trump's word cloud
    ax2.imshow(trump_03_11_positive_wordcloud, interpolation='bilinear')
    ax2.axis('off')  # Hide axes
    ax2.set_title("Trump's Positive", fontsize=16)

    # Show the plot in Streamlit
    st.pyplot(positive_03_11_fig)

    # Create side-by-side Negative word clouds
    negative_03_11_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # Display Biden's word cloud
    ax1.imshow(biden_03_11_negative_wordcloud, interpolation='bilinear')
    ax1.axis('off')  # Hide axes
    ax1.set_title("Biden's Negative", fontsize=16)

    # Display Trump's word cloud
    ax2.imshow(trump_03_11_negative_wordcloud, interpolation='bilinear')
    ax2.axis('off')  # Hide axes
    ax2.set_title("Trump's Negative", fontsize=16)

    # Show the plot in Streamlit
    st.pyplot(negative_03_11_fig)


    # 03/11/20 Hashtag Word Cloud
    # Generate word clouds for both candidates
    biden_03_11_hashtag_negative_wordcloud = generate_wordcloud('biden_03_11_hashtag_negative_wordcloud.csv')
    trump_03_11_hashtag_negative_wordcloud = generate_wordcloud('trump_03_11_hashtag_negative_wordcloud.csv')
    
    # Create side-by-side Negative word clouds
    negative_hashtag_03_11_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # Display Biden's word cloud
    ax1.imshow(biden_03_11_hashtag_negative_wordcloud, interpolation='bilinear')
    ax1.axis('off')  # Hide axes
    ax1.set_title("Biden's Negative Hashtag", fontsize=16)

    # Display Trump's word cloud
    ax2.imshow(trump_03_11_hashtag_negative_wordcloud, interpolation='bilinear')
    ax2.axis('off')  # Hide axes
    ax2.set_title("Trump's Negative Hashtag", fontsize=16)

    # Show the plot in Streamlit
    st.pyplot(negative_hashtag_03_11_fig)


    # Word Cloud for the date of which each candidate had their own lowest polarity mean
    st.subheader("Lowest Polarity Mean Dates")    
    # Generate filtered word clouds for both candidates
    biden_min_pol_date_negative_wordcloud_df = generate_wordcloud('biden_min_pol_date_negative_wordcloud.csv')
    trump_min_pol_date_negative_wordcloud_df = generate_wordcloud('trump_min_pol_date_negative_wordcloud.csv')

    # Create side-by-side Positive word clouds
    min_pol_date_negative_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # Display Both candidates' min polarity date word cloud
    ax1.imshow(biden_min_pol_date_negative_wordcloud_df, interpolation='bilinear')
    ax1.axis('off')  # Hide axes
    ax1.set_title("Biden's 15/10/20 Negative", fontsize=16)

    # Display Trump's word cloud
    ax2.imshow(trump_min_pol_date_negative_wordcloud_df, interpolation='bilinear')
    ax2.axis('off')  # Hide axes
    ax2.set_title("Trump's 21/10/20 Negative", fontsize=16)

    # Show the plot in Streamlit
    st.pyplot(min_pol_date_negative_fig)


    # Min Polarity Date Hashtag Word Cloud
    # Generate word clouds for both candidates
    biden_min_pol_date_hashtag_negative_wordcloud = generate_wordcloud('biden_min_pol_date_negative_hashtag_wordcloud.csv')
    trump_min_pol_date_hashtag_negative_wordcloud = generate_wordcloud('trump_min_pol_date_negative_hashtag_wordcloud.csv')
    
    # Create side-by-side Negative word clouds
    negative_hashtag_min_pol_date_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # Display Biden's word cloud
    ax1.imshow(biden_min_pol_date_hashtag_negative_wordcloud, interpolation='bilinear')
    ax1.axis('off')  # Hide axes
    ax1.set_title("Biden's Negative Hashtag", fontsize=16)

    # Display Trump's word cloud
    ax2.imshow(trump_min_pol_date_hashtag_negative_wordcloud, interpolation='bilinear')
    ax2.axis('off')  # Hide axes
    ax2.set_title("Trump's Negative Hashtag", fontsize=16)

    # Show the plot in Streamlit
    st.pyplot(negative_hashtag_min_pol_date_fig)

# Footer
st.markdown("This dashboard provides insights into the Twitter election data for the 2020 U.S. Presidential election.")