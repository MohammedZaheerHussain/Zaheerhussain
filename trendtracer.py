import pandas as pd
import numpy as np
from dateutil import parser
import isodate

# Data visualization libraries
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

sns.set(style="darkgrid", color_codes=True)
from sklearn.feature_extraction.text import CountVectorizer

# Google API
from googleapiclient.discovery import build
import warnings

warnings.filterwarnings("ignore", message="Matplotlib currently does not support Devanagari natively.")

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import (word_tokenize)

nltk.download('stopwords')
nltk.download('punkt')
from wordcloud import WordCloud

api_key = 'AIzaSyA1wEEOxx2TljBrn4ujboIgqD3q6hZYdVg'
channel_ids = [
    # 'UCZyCposXwcyopaACep44maQ',  # Alex Costa
    # 'UCHdPuKshPHyRHE3rz75rXdw',  # Dre Drexler
     'UCjcqzy7MSaN2KPnzOKIcpEQ',  # DQ salmaan
     'UCOe1D_N5alGASGi47tDTGGg',  # Pat Cummins
    # 'UChXqvaGDLD8jfjFXPq66XHg',  # Akshanshu Aswal
    # 'UCqgdGSJPEKCGwElnwnKGE6Q',  # Mohit Sharma
     #'UCbq8_4_mFAx_rzDF5VT7MJw',  # Blumaan
     'UC64guZp8DXzqrIQ5OPedviw',  # HoomanTV
    # 'UC0QHWhjbe5fGJEPz3sVb6nw',  # Doctor Mike
     'UCHWbZM3BIGgZksvXegx_h3w',  # Enes yilmazer

]

youtube = build('youtube', 'v3', developerKey=api_key)


def get_channel_stats(youtube, channel_ids):
    all_data = []
    request = youtube.channels().list(
        part='snippet,contentDetails,statistics',
        id=','.join(channel_ids))
    response = request.execute()

    for i in range(len(response['items'])):
        data = dict(channelName=response['items'][i]['snippet']['title'],
                    subscribers=response['items'][i]['statistics']['subscriberCount'],
                    views=response['items'][i]['statistics']['viewCount'],
                    totalVideos=response['items'][i]['statistics']['videoCount'],
                    playlistId=response['items'][i]['contentDetails']['relatedPlaylists']['uploads'])

        all_data.append(data)

    return pd.DataFrame(all_data)


def get_video_ids(youtube, playlist_id):
    video_ids = []
    request = youtube.playlistItems().list(
        part='contentDetails',
        playlistId=playlist_id,
        maxResults=50)
    response = request.execute()

    for i in range(len(response['items'])):
        video_ids.append(response['items'][i]['contentDetails']['videoId'])

    next_page_token = response.get('nextPageToken')
    more_pages = True

    while more_pages:
        if next_page_token is None:
            more_pages = False
        else:
            request = youtube.playlistItems().list(
                part='contentDetails',
                playlistId=playlist_id,
                maxResults=50,
                pageToken=next_page_token)
            response = request.execute()

            for i in range(len(response['items'])):
                video_ids.append(response['items'][i]['contentDetails']['videoId'])

            next_page_token = response.get('nextPageToken')

    return video_ids


def get_video_details(youtube, video_ids):
    video_df = []

    for i in range(0, len(video_ids), 50):
        request = youtube.videos().list(
            part="snippet,contentDetails,statistics",
            id=','.join(video_ids[i:i + 50])
        )
        response = request.execute()

        for video in response['items']:
            stats_to_keep = {
                'snippet': ['channelTitle', 'title', 'description', 'tags', 'publishedAt'],
                'statistics': ['viewCount', 'likeCount', 'favoriteCount', 'commentCount'],
                'contentDetails': ['duration', 'definition', 'caption']
            }
            video_info = {'video_id': video['id'], 'video_name': video['snippet']['title']}  # Include video name
            for k, v_list in stats_to_keep.items():
                for v in v_list:
                    try:
                        video_info[v] = video[k][v]
                    except KeyError:
                        video_info[v] = None
            video_df.append(video_info)

    return pd.DataFrame(video_df)


def get_comments_in_videos(youtube, video_ids):
    all_comments = []

    for video_id in video_ids:
        try:
            request = youtube.commentThreads().list(
                part="snippet,replies",
                videoId=video_id
            )
            response = request.execute()

            comments_in_video = [comment['snippet']['topLevelComment']['snippet']['textOriginal']

                                 for comment in response['items'][0:10]]
            comments_in_video_info = {'video_id': video_id, 'comments': comments_in_video}

            all_comments.append(comments_in_video_info)


        except Exception as e:
            if 'commentsDisabled' in str(e):
                print(f'Comments are disabled for video {video_id}.')
            else:
                # Log or store more detailed information about the error
                print(f'Could not get comments for video {video_id}. Error: {str(e)}')

    return pd.DataFrame(all_comments)


channel_data = get_channel_stats(youtube, channel_ids)

# Convert count columns to numeric columns
numeric_cols = ['subscribers', 'views', 'totalVideos']
channel_data[numeric_cols] = channel_data[numeric_cols].apply(pd.to_numeric, errors='coerce')
sns.set(rc={'figure.figsize': (10, 8)})

plt.figure(figsize=(10, 8))  # Adjust the figure size if needed
ax_subscribers = sns.barplot(x='channelName', y='subscribers',
                             data=channel_data.sort_values('subscribers', ascending=False))
ax_subscribers.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x / 1000) + 'K'))
ax_subscribers.set_xticks(ax_subscribers.get_xticks())
ax_subscribers.set_xticklabels(ax_subscribers.get_xticklabels(), rotation=90, ha="right")
ax_subscribers.set_title('Subscribers')

# Bar plot for views
plt.figure(figsize=(10, 8))  # Adjust the figure size if needed
ax_views = sns.barplot(x='channelName', y='views', data=channel_data.sort_values('views', ascending=False))
ax_views.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x / 1000) + 'K'))
ax_views.set_xticks(ax_views.get_xticks())
ax_views.set_xticklabels(ax_views.get_xticklabels(), rotation=90, ha="right")
ax_views.set_title('Views')

plt.show()

# Create a dataframe with video statistics and comments from all channels
video_df = []  # Initialize an empty list for video DataFrames
comments_df = []  # Initialize an empty list for comment DataFrames

for c in channel_data['channelName'].unique():
    print("Getting video information from channel: " + c)
    playlist_id = channel_data.loc[channel_data['channelName'] == c, 'playlistId'].iloc[0]
    video_ids = get_video_ids(youtube, playlist_id)

    # Get video data
    video_data = get_video_details(youtube, video_ids)
    # Create a new DataFrame for each channel
    new_video_df = pd.DataFrame(video_data)
    new_video_df['publishedAt'] = pd.to_datetime(new_video_df['publishedAt'], errors='coerce')
    # Append to the list
    video_df.append(new_video_df)

    # Get comment data
    comments_data = get_comments_in_videos(youtube, video_ids)
    # Create a new DataFrame for each set of comments
    new_comments_df = pd.DataFrame(comments_data)
    # Append to the list
    comments_df.append(new_comments_df)

# Concatenate all DataFrames from the lists outside the for loop
video_df = pd.concat(video_df, ignore_index=True)

comments_df = pd.concat(comments_df, ignore_index=True)
# Save comments_df to a CSV file
comments_df.to_csv('youtube_comments.csv', index=False)

# Additional processing on video_df
video_df['publishDayName'] = video_df['publishedAt'].apply(lambda x: x.strftime("%A"))

# Write video data to CSV file for future references
video_df.to_csv('video_data_top10_channels.csv', index=False)

# Check for null values
video_df.isnull().any()
# Concatenate all DataFrames from the lists outside the for loop
# Concatenate all DataFrames from the lists outside the for loop
video_df = pd.concat([video_df], ignore_index=True)
comments_df = pd.concat([comments_df], ignore_index=True)

# Display value counts for 'publishedAt'
video_df['publishedAt'].sort_values().value_counts()

# Convert 'publishedAt' to datetime
video_df['publishedAt'] = pd.to_datetime(video_df['publishedAt'], errors='coerce')

# Convert 'duration' to seconds
video_df['durationSecs'] = video_df['duration'].apply(lambda x: isodate.parse_duration(x))
video_df['durationSecs'] = video_df['durationSecs'].astype('timedelta64[s]')

# Add number of tags
video_df['tagsCount'] = video_df['tags'].apply(lambda x: 0 if x is None else len(x))

# Title character length
video_df['titleLength'] = video_df['title'].apply(lambda x: len(x))

# Filter the data to only include videos with more than 100,000 views
mask = pd.to_numeric(video_df['viewCount']) > 100000  # Ensure 'viewCount' is numeric
video_df['viewCount'] = pd.to_numeric(video_df['viewCount'], errors='coerce')

# Filter based on viewCount threshold
mask = video_df['viewCount'] > 100000

# Set up the figure size
plt.figure(figsize=(10, 8))

# Create the bar plot with adjusted parameters
ax = sns.barplot(
    x="channelTitle",
    y="viewCount",
    data=video_df.loc[mask],
    hue="channelTitle",
    palette="pastel",
    ci=None  # Disable error bars
)

# Set the title
plt.title('Views per channel (filtered)', fontsize=14)

# Display the plot with rotated x-axis labels
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Display the plot
plt.show()
# Convert 'viewCount' to numeric, handle non-numeric values by coercing to NaN
video_df['viewCount'] = pd.to_numeric(video_df['viewCount'], errors='coerce')
# Group the data by channel and plot the median view count for each channel
df_agg = video_df.groupby('channelTitle')['viewCount'].agg(['median', 'count'])
df_agg = df_agg[df_agg['count'] > 10]  # only include channels with more than 10 videos
df_agg = df_agg.sort_values('median', ascending=False)[:20]  # only include the top 20 channels by median view count
# Instead of using palette directly, assign x to hue
sns.barplot(x=df_agg.index, y=df_agg['median'], palette='pastel')
plt.title('Top 10 channels by median view count', fontsize=14)
plt.xticks(rotation=90)
plt.show()

fig, ax = plt.subplots(1, 2)
sns.scatterplot(data=video_df, x="commentCount", y="viewCount", ax=ax[0])
sns.scatterplot(data=video_df, x="likeCount", y="viewCount", ax=ax[1])

# Convert timedelta values to seconds for plotting
video_df['durationSecs'] = video_df['durationSecs'].dt.total_seconds()

# Convert 10000 seconds to timedelta
threshold_duration = pd.to_timedelta(10000, unit='s')

# Plot histogram with duration less than the threshold
sns.histplot(data=video_df[video_df['durationSecs'] < threshold_duration.total_seconds()], x="durationSecs", bins=30)
plt.hexbin(x=video_df["titleLength"], y=video_df["viewCount"], gridsize=30, cmap="Blues")
plt.colorbar()

stop_words = set(stopwords.words('english'))
video_df['title_no_stopwords'] = video_df['title'].apply(
    lambda x: [item for item in str(x).split() if item not in stop_words])

all_words = list([a for b in video_df['title_no_stopwords'].tolist() for a in b])
all_words_str = ' '.join(all_words)


def plot_cloud(wordcloud):
    plt.figure(figsize=(10, 8))
    plt.imshow(wordcloud)
    plt.axis("off")


wordcloud = WordCloud(width=2000, height=1000, random_state=1, background_color='black',
                      colormap='viridis', collocations=False).generate(all_words_str)
plot_cloud(wordcloud)

# Assuming 'video_df' is your DataFrame
video_df['publishDayName'] = video_df['publishedAt'].apply(lambda x: x.strftime("%A"))

# Define the order of weekdays
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Create a countplot for 'publishDayName'
sns.set(style="whitegrid")
plt.figure(figsize=(10, 8))
sns.countplot(x='publishDayName', data=video_df, order=weekdays, palette="viridis")

plt.title('Number of Videos Published by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Number of Videos Published')
plt.show()

stop_words = set(stopwords.words('english'))

comments_df['comments_no_stopwords'] = comments_df['comments'].apply(
    lambda x: [item for item in str(x).split() if item not in stop_words])

all_words = list([a for b in comments_df['comments_no_stopwords'].tolist() for a in b])
all_words_str = ' '.join(all_words)

wordcloud = WordCloud(width=2000, height=1000, random_state=1, background_color='black',
                      colormap='viridis', collocations=False).generate(all_words_str)
plot_cloud(wordcloud)

import warnings

warnings.filterwarnings("ignore")


def get_video_details(youtube, video_ids):
    all_video_stats = []

    for i in range(0, len(video_ids), 50):
        request = youtube.videos().list(
            part='snippet,statistics',
            id=','.join(video_ids[i:i + 50]))
        response = request.execute()

        for video in response['items']:
            video_stats = dict(Title=video['snippet']['title'],
                               Published_date=video['snippet']['publishedAt'],
                               Views=video['statistics'].get('viewCount'),
                               Likes=video['statistics'].get('likeCount'),
                               Dislikes=video['statistics'].get('dislikeCount', 0),
                               Comments=video['statistics'].get('commentCount', 0)
                               )
            all_video_stats.append(video_stats)

    return all_video_stats


video_details = get_video_details(youtube, video_ids)

video_data = pd.DataFrame(video_details)
video_data['Published_date'] = pd.to_datetime(video_data['Published_date']).dt.date
video_data['Views'] = pd.to_numeric(video_data['Views'])
video_data['Likes'] = pd.to_numeric(video_data['Likes'])
video_data['Dislikes'] = pd.to_numeric(video_data['Dislikes'])
video_data['Views'] = pd.to_numeric(video_data['Views'])

top10_videos = video_data.sort_values(by='Views', ascending=False).head(10)
top10_videos

# Assuming 'video_data' is your DataFrame
video_data['Month'] = pd.to_datetime(video_data['Published_date']).dt.strftime('%b')
videos_per_month = video_data.groupby('Month').size().reset_index(name='Count')

# Define the order of months
sort_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
videos_per_month['Month'] = pd.Categorical(videos_per_month['Month'], categories=sort_order, ordered=True)
videos_per_month = videos_per_month.sort_values('Month')

# Plotting
plt.figure(figsize=(10, 8))
ax2 = sns.barplot(x='Month', y='Count', data=videos_per_month, palette="viridis")

plt.title('Number of Videos Published per Month')
plt.xlabel('Month')
plt.ylabel('Number of Videos Published')
plt.show()

import pandas as pd
import numpy as np

pd.set_option('display.max_colwidth', 200)

# Visualization packages
import matplotlib.pyplot as plt

# NLP packages
from textblob import TextBlob

import warnings

warnings.filterwarnings("ignore")

TextBlob("The movie is good").sentiment

# Importing YouTube comments data
data = pd.read_csv('youtube_comments.csv', encoding='utf8')

# Check the number of rows in the DataFrame
num_rows = data.shape[0]

# Specify the desired sample size
sample_size = min(2000, num_rows)  # Choose the smaller of 2000 and the number of rows

# Extracting a sample with replacement
comm = data.sample(sample_size, replace=True)

# Print the columns of the DataFrame to check if 'Comment' exists
print(comm.columns)

# Calculating the Sentiment Polarity
comm['pol'] = comm['comments'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

# Assigning sentiment polarity values to categories
comm['pol'] = comm['pol'].apply(lambda x: 'Positive' if x > 0 else ('Neutral' if x == 0 else 'Negative'))

# Separate DataFrames for positive, negative, and neutral comments
df_positive = comm[comm['pol'] == 'Positive']
df_negative = comm[comm['pol'] == 'Negative']
df_neutral = comm[comm['pol'] == 'Neutral']

# Displaying the first 10 rows of positive, negative, and neutral comments
print("Positive Comments:")
print(df_positive.head(10))
print("\nNegative Comments:")
print(df_negative.head(10))
print("\nNeutral Comments:")
print(df_neutral.head(10))

# Plotting the bar chart for sentiment distribution
plt.figure(figsize=(10, 8))
comm['pol'].value_counts().sort_index().plot(kind='bar', color=['green', 'red', 'grey'])
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# Plotting the pie chart for sentiment distribution
plt.figure(figsize=(10, 8))
comm['pol'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['green', 'red', 'grey'])
plt.title('Sentiment Distribution')
plt.show()

video_data['Views'] = video_data['Views'].astype(float)
video_data['Likes'] = video_data['Likes'].astype(float)
video_data['Dislikes'] = video_data['Dislikes'].astype(float)
video_data['Comments'] = video_data['Comments'].astype(float)

plt.figure(figsize=(10, 8))
sns.lineplot(x=video_data['Published_date'], y=video_data['Views'], label='Views')
sns.lineplot(x=video_data['Published_date'], y=video_data['Likes'], label='Likes')
sns.lineplot(x=video_data['Published_date'], y=video_data['Dislikes'], label='Dislikes')
sns.lineplot(x=video_data['Published_date'], y=video_data['Comments'], label='Comments')
plt.title('Engagement Over Time')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend()
plt.show()

import matplotlib.pyplot as plt
import pandas as pd
from textblob import TextBlob


# Function to generate n-grams
def generate_ngrams(text, n):
    words = text.split()
    ngrams = [words[i:i + n] for i in range(len(words) - n + 1)]
    return [' '.join(ngram) for ngram in ngrams]


# Function to plot n-grams
def plot_ngrams(ngrams, title):
    plt.figure(figsize=(10, 8))
    ngrams_list = list(ngrams)  # Convert the zip object to a list
    plt.barh(range(len(ngrams_list)), [freq for gram, freq in ngrams_list])
    plt.yticks(range(len(ngrams_list)), [gram for gram, freq in ngrams_list])
    plt.xlabel('Frequency')
    plt.ylabel('N-gram')
    plt.title(title)
    plt.gca().invert_yaxis()  # Invert y-axis to have the highest frequency at the top
    plt.show()


# Function to plot sentiment distribution
def plot_sentiment_distribution(sentiments, title):
    plt.figure(figsize=(10, 8))
    sentiments.value_counts().plot(kind='bar', color=['green', 'red', 'yellow'])
    plt.title(title)
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.show()


# Generate n-grams
unigrams = generate_ngrams(' '.join(comm['comments']), 1)
bigrams = generate_ngrams(' '.join(comm['comments']), 2)
trigrams = generate_ngrams(' '.join(comm['comments']), 3)

# Plot top 10 n-grams
unigram_freq = pd.Series(unigrams).value_counts().head(10)  # Top 10 unigrams
bigram_freq = pd.Series(bigrams).value_counts().head(10)  # Top 10 bigrams
trigram_freq = pd.Series(trigrams).value_counts().head(10)  # Top 10 trigrams
plot_ngrams(unigram_freq.items(), 'Top 10 Unigrams')
plot_ngrams(bigram_freq.items(), 'Top 10 Bigrams')
plot_ngrams(trigram_freq.items(), 'Top 10 Trigrams')

# Aggregate data
video_stats = video_data.groupby('Title').agg({
    'Views': 'sum',
    'Likes': 'sum',
    'Dislikes': 'sum',
    'Comments': 'sum'
}).reset_index()

# Calculate engagement rate
video_stats['Engagement'] = video_stats['Likes'] + video_stats['Dislikes'] + video_stats['Comments']

# Merge with channel data to get subscribers
video_stats = pd.merge(video_stats, channel_data[['channelName', 'subscribers']], left_on='Title',
                       right_on='channelName')

# Calculate views per subscriber
video_stats['Views_Per_Subscriber'] = video_stats['Views'] / video_stats['subscribers']

# Define trending criteria
trending_threshold = 10000  # Adjust as needed

# Identify trending videos
trending_videos = video_stats[
    (video_stats['Views'] > trending_threshold) &
    (video_stats['Engagement'] > trending_threshold) &
    (video_stats['Views_Per_Subscriber'] > 0.1)  # Example criterion
    ]

# Visualize trending videos
plt.figure(figsize=(10, 8))
sns.barplot(x='Title', y='Views', data=trending_videos, palette='viridis')
plt.title('Trending Videos')
plt.xlabel('Video Title')
plt.ylabel('Views')
plt.xticks(rotation=45, ha='right')
plt.show()

import pandas as pd
from textblob import TextBlob


# Function to perform sentiment analysis on comments
def analyze_sentiment(comments):
    sentiment_labels = []
    for comment in comments:
        # Perform sentiment analysis using TextBlob
        blob = TextBlob(str(comment))
        polarity = blob.sentiment.polarity
        if polarity > 0:
            sentiment_labels.append('Positive')
        elif polarity < 0:
            sentiment_labels.append('Negative')
        else:
            sentiment_labels.append('Neutral')
    return sentiment_labels


# Assuming you have a DataFrame 'comments_df' with a column 'comments' containing the comments
comments_df['sentiment'] = analyze_sentiment(comments_df['comments'])

# Save the DataFrame to a CSV file
comments_df.to_csv('comments_with_sentiment.csv', index=False)
