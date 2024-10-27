import pandas as pd
import plotly.express as px
import os
import re
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import streamlit as st

st.title("Twitter Analysis: Top US Cities and Hashtag Analysis for Each Candidate")
file_path = '/Users/may/Documents/GitHub/DataCircle_Twitter_Project/twitter_cleaned_data.csv'

if os.path.exists(file_path):
    df_combined = pd.read_csv(file_path, lineterminator='\n')

    # -------------------- Top US Cities by Tweet Count --------------------
    # Filter for tweets from U.S. cities
    us_tweets = df_combined[df_combined['country'].str.strip().str.lower() == 'united states']

    # Check if there are tweets from the US
    if not us_tweets.empty:
        incorrect_city_names = ['unknown', 'unkown', 'unknwn', 'unlown', 'unknownn']
        #droppig uknown values 
        us_tweets = us_tweets[~us_tweets['city'].str.strip().str.lower().isin(incorrect_city_names)]

        if not us_tweets.empty:
            # Count tweets by city
            tweets_count_by_city = us_tweets.groupby('city').size().reset_index(name='Tweet Count')
            
            # Get the top 5 cities with the maximum tweets
            top_5_cities = tweets_count_by_city.nlargest(5, 'Tweet Count')

            # Display the top cities
            st.write("Top US Cities by Tweet Count (excluding incorrect city names):")
            st.dataframe(top_5_cities)

            # creating a bar chart
            fig = px.bar(top_5_cities,
                         x='city',
                         y='Tweet Count',
                         title='Top 5 US Cities by Number of Tweets',
                         labels={'Tweet Count': 'Number of Tweets', 'city': 'City'},
                         color='Tweet Count',
                         color_continuous_scale=px.colors.sequential.Viridis)

            st.plotly_chart(fig)
        else:
            st.write("No valid city data available after filtering out incorrect city names.")
    else:
        st.write("No tweets found from the United States.")
    
    # -------------------- Hashtag Analysis for Each Candidate --------------------
    # Function to extract hashtags from text
    def extract_hashtags(text):
        return re.findall(r"#(\w+)", str(text))

    # Add a new column with hashtags
    us_tweets['hashtags'] = us_tweets['tweet_cleaned\r'].apply(extract_hashtags)

    # Filter tweets for each candidate
    trump_tweets = us_tweets[us_tweets['candidate'].str.strip().str.lower() == 'trump']
    biden_tweets = us_tweets[us_tweets['candidate'].str.strip().str.lower() == 'biden']

    # Function to get top hashtags for a candidate
    def get_top_hashtags(tweets, top_n=10):
        hashtags = Counter([ht.lower() for tags in tweets['hashtags'] for ht in tags])
        return hashtags.most_common(top_n)

    # Get popular hashtags for Trump and Biden
    trump_top_hashtags = get_top_hashtags(trump_tweets)
    biden_top_hashtags = get_top_hashtags(biden_tweets)

    # Prepare data for visualization
    trump_data = pd.DataFrame(trump_top_hashtags, columns=['Hashtag', 'Count'])
    trump_data['Candidate'] = 'Trump'

    biden_data = pd.DataFrame(biden_top_hashtags, columns=['Hashtag', 'Count'])
    biden_data['Candidate'] = 'Biden'

    # Combine data for both candidates
    combined_data = pd.concat([trump_data, biden_data])

    # Visualize top hashtags for each candidate
    fig = px.bar(combined_data, 
                 x='Hashtag', 
                 y='Count', 
                 color='Candidate',
                 title='Top Hashtags for Each Candidate',
                 labels={'Hashtag': 'Hashtag', 'Count': 'Count', 'Candidate': 'Candidate'},
                 color_discrete_map={'Trump': 'red', 'Biden': 'blue'})  # Red for Trump, Blue for Biden

    fig.update_layout(barmode='group', xaxis_title="Hashtags", yaxis_title="Count")
    st.plotly_chart(fig)

    # -------------------- Word Cloud for Hashtags in New York --------------------
    # Filter for tweets from New York
    nyc_tweets = df_combined[(df_combined['city'].str.strip().str.lower() == 'new york')]

    # Separate hashtags for each candidate in New York
    trump_nyc_hashtags = nyc_tweets[nyc_tweets['candidate'].str.lower() == 'trump']['tweet_cleaned\r'].apply(extract_hashtags).sum()
    biden_nyc_hashtags = nyc_tweets[nyc_tweets['candidate'].str.lower() == 'biden']['tweet_cleaned\r'].apply(extract_hashtags).sum()

    # Generate word clouds for each candidate from New York
    st.write("Trump Hashtags Word Cloud")
    trump_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(trump_nyc_hashtags))
    fig_trump, ax_trump = plt.subplots()
    ax_trump.imshow(trump_wordcloud, interpolation='bilinear')
    ax_trump.axis('off')
    st.pyplot(fig_trump)

    st.write("Biden Hashtags Word Cloud")
    biden_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(biden_nyc_hashtags))
    fig_biden, ax_biden = plt.subplots()
    ax_biden.imshow(biden_wordcloud, interpolation='bilinear')
    ax_biden.axis('off')
    st.pyplot(fig_biden)

else:
    st.write("File not found.")



