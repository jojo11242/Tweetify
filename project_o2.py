
from sqlalchemy import create_engine, types
from urllib.parse import quote
from sqlalchemy.sql import text

# scape data 
import pandas as pd
# from Scweet_master.Scweet.scweet import scrape
# import tools.tweet_analysis as tw
# import tools.AzureSQL_DDL as az
# import tools.preprocessing as pp
from datetime import datetime
# import time
import snscrape.modules.twitter as sntwitter
import re


# def init_question():
#     num_kw = int(input('How many keywords you would like to compare?: '))
#     kw = ''
#     for i in range(0, num_kw):
#         tem = input(f'Please input keyword {i+1}: ')
#         kw = kw + tem + ','
#     kw = kw[:-1]
#     kw_ls = kw.split(',')
#     since = input('Which is the start date of tweets? e.g. 2022-06-01: ')
#     pattern = re.compile(r'\d{4}-\d{2}-\d{2}')
#     assert pattern.match(since), print(
#         '!!Please input the date in the format of YYYY-MM-DD!!')
#     until = input('Which is the end date of tweets? e.g. 2022-11-21: ')
#     assert pattern.match(until), print(
#         '!!Please input the date in the format of YYYY-MM-DD!!')
#     return kw_ls, num_kw, since, until


# def query(text, since, until):
#     q = text  # keyword
#     q += f" until:{until}"
#     q += f" since:{since}"
#     # q += ' geocode:41.4925374,-99.9018131,1500km'
#     return q


# def snscraperper(text, since, until, interval=3): #perper
#     d = interval
#     tweet_list = []

#     # create date list with specific interval s
#     dt_rng = pd.date_range(start=since, end=until, freq=f'{d}D')

#     # Scrape for each day
#     for dt in dt_rng:
#         # since to until = since + 1 day
#         q = query(text, since=datetime.strftime(dt, '%Y-%m-%d'),
#                   until=datetime.strftime(dt+pd.to_timedelta(1, 'D'), '%Y-%m-%d'))
#         print('start scraping {date}'.format(
#             date=datetime.strftime(dt, '%Y-%m-%d')))

#         counter = 0
#         try:
#             for i, tweet in enumerate(sntwitter.TwitterSearchScraper(q).get_items()):
#                 tweet_list.append([tweet.date, tweet.user.username, tweet.rawContent,  tweet.likeCount,
#                                   tweet.replyCount, tweet.retweetCount, tweet.quoteCount, tweet.url])
#                 counter += 1
#                 if counter % 100 == 0:
#                     print(f'{counter} scrapped')

#             print('finished scraping {date}, # of tweets: {no_tweet}'.format(
#                 date=datetime.strftime(dt, '%Y-%m-%d'), no_tweet=counter))
#         except:
#             print('error occured in {date}'.format(
#                 date=datetime.strftime(dt, '%Y-%m-%d')))
#             continue

#     # Creating a dataframe from the tweets list above
#     tweets_df = pd.DataFrame(tweet_list, columns=[
#                              'Timestamp', 'Username', 'Embedded_text', 'Likes', 'Comments', 'Retweets', 'Quotes', 'Tweet URL'])
#     return tweets_df

# kw_ls, num_kw, since, until = init_question()


# for i in range(num_kw):
#     tweets_df = snscraperper(kw_ls[i], since, until)
#     filename = f'df_sns{i+1}.csv'
#     tweets_df.to_csv(filename)

kw_ls=['disney california','unistudios']
#data cleaning 

dfs = [pd.read_csv(f"df_sns{i+1}.csv") for i in range(len(kw_ls))]

def change_data_type(df):
    try:
        df.columns = df.columns.str.lower()
        data_types = {'timestamp': 'datetime64[ns]', 'username': 'object', 
                      'embedded_text': 'object', 'likes': 'int32', 
                      'comments': 'int32', 'retweets': 'int32'}
        df = df.astype(data_types)
        return df
    except Exception as e:
        print(f'An error occurred while converting the data types: {e}')
        return None

for i, df in enumerate(dfs):
    df = df.drop(columns=['Unnamed: 0','Quotes', 'Tweet URL'], axis=1)
    change_data_type(df)
    df.to_csv(f"{kw_ls[i].lower().replace(' ', '_')}.csv", index=False)


tweets_df = []
for kw in kw_ls:
    filename = f"{kw.lower().replace(' ', '_')}.csv"
    tweets = pd.read_csv(filename)
    tweets['brands'] = kw
    tweets_df.append(tweets)

tweets = pd.concat(tweets_df)
# process the combined tweets DataFrame here

tweets.to_csv('brands_file.csv', index=False)



import os
import psycopg2

#create a connection to the database
password = quote('password')
engine = create_engine(f'postgresql://postgres:{password}@localhost:5432/twitter')
tweets.to_sql('tweets', engine, if_exists='replace')


from textblob import TextBlob

# Retrieve the embedded_text and brands columns from the database
query = "select embedded_text, timestamp, brands, likes, retweets from tweets"
connection = engine.connect()
tweets = pd.read_sql(text(query), con=connection)

#Perform sentiment analysis on the embedded_text column
tweets['embedded_text'] = tweets['embedded_text'].fillna('')

def textblob_polarity(text):
    text = str(text)
    return TextBlob(text).sentiment.polarity

tweets['polarity'] = tweets['embedded_text'].apply(textblob_polarity)


# print the polarity values
print(tweets['polarity'])
print(type(tweets))
print(tweets.keys())



def getAnalysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'

# Add the result column to the DataFrame
tweets['result'] = tweets['polarity'].apply(getAnalysis)
print(tweets)
# # Save the updated DataFrame to a csv file if desired
tweets.to_csv('result3_tweets.csv', index=False)


a = 0
positive_count=0
negative_count=0
neutral_count=0


for i in tweets['embedded_text']:
    a = a + textblob_polarity(i)
    if textblob_polarity(i) > 0:
        positive_count += 1

    elif textblob_polarity(i) < 0:
        negative_count += 1
    else:
         neutral_count += 1


         text_count = positive_count + negative_count + neutral_count
print(text_count)

p_positive = (positive_count/text_count)*100
print(p_positive)

p_negative = (negative_count/text_count)*100
print(p_negative)

p_neutral = (neutral_count/text_count)*100
print(p_neutral)

print(tweets.columns)


# need to seperate by brands
#nltk
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd


## Load the tweets into a pandas DataFrame
tweets = pd.read_csv("result3_tweets.csv")
nltk.download('punkt')
nltk.download('stopwords')


# Remove stop words
stop_words = set(stopwords.words("english"))
stop_words.update(['disneyland', 'universialstudios', 'disney', 'reply', 'like'])

# Remove special characters, symbols, and numbers
def extract_words(text):
    words = word_tokenize(text)
    filtered_words = []
    for word in words:
        word = re.sub(r'[^a-zA-Z]', '', word)
        if word.isalpha():
            filtered_words.append(word.lower())
    return filtered_words

negative_tweets = tweets[tweets['result'] == 'Negative']
negative_words = []
for i in negative_tweets['embedded_text']:
    for word in extract_words(i):
        if word not in stop_words:
            negative_words.append(word)
negative_list = nltk.FreqDist(negative_words)

# Print the most common positive words
print("Most common words in negative tweets:")
print(negative_list.most_common(30))

positive_tweets = tweets[tweets['result'] == 'Positive']
positive_words = []
for i in positive_tweets['embedded_text']:
    for word in extract_words(i):
        if word not in stop_words:
            positive_words.append(word)
positive_list = nltk.FreqDist(positive_words)

print("Most common words in positive tweets:")
print(positive_list.most_common(30))



from wordcloud import WordCloud
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\victorlee\Desktop\my_python\JDE\project\result3_tweets.csv")

# Combine all the tweets into a single string
all_words = ' '.join(df['embedded_text'])

# Generate the word cloud
cloud = WordCloud(width=500, height=300, random_state=0, max_font_size=100).generate(all_words)

# Plot the word cloud
plt.imshow(cloud)
plt.show()


#====Scattor plot

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
import random

def remove_outliers(data):
    z = np.abs(stats.zscore(data))
    threshold = 1
    return data[(z < threshold)]

kw_ls=['disney california','unistudios']

# Load the data from the csv file
df = pd.read_csv(r'C:\Users\victorlee\Desktop\my_python\JDE\project\result3_tweets.csv')
print(df.head(20))
# Convert the timestamp column to datetime format and extract the date
df['timestamp_2'] = pd.to_datetime(df['timestamp'])
df['datestamp'] = df['timestamp_2'].dt.date

# Filter data for the year 2022 only
df = df[(df['timestamp_2'] >= '2022-01-01') & (df['timestamp_2'] < '2023-02-11')]

# Group the data by date and brand, and count the sum of each date per brand
df_count = df.groupby(['datestamp', 'brands'])['likes', 'retweets'].sum().reset_index()

# Mapping of keywords to colors for plotting
color_mapping = {kw: f'C{i}' for i, kw in enumerate(kw_ls)}

fig, ax = plt.subplots(figsize=(20, 10))

# Loop through each keyword in the list
for kw in kw_ls:
    # Extract the data for each brand
    data = df_count[df_count['brands'] == kw].set_index('datestamp')

    # Remove outliers
    data['likes'] = remove_outliers(data['likes'])
    data['retweets'] = remove_outliers(data['retweets'])

    # Plot the scatter plot for likes and retweets for each brand
    ax.scatter(data.index, data['likes'], c=color_mapping[kw], label=kw.capitalize() + ' Likes')
    ax.scatter(data.index, data['retweets'], c=color_mapping[kw], marker='^', label=kw.capitalize() + ' Retweets')

# Add the legend and labels
ax.legend(loc='upper left')
ax.set_xlabel('Timeframe')
ax.set_ylabel('Number of Likes and Retweets per interval month')
ax.set_title('Brand Comparison: Likes and Retweets')
plt.show()


# ====line graph

import matplotlib.pyplot as plt
import pandas as pd

# Load the data from the csv file
df = pd.read_csv(r'C:\Users\victorlee\Desktop\my_python\JDE\project\result3_tweets.csv')

# Convert the timestamp column to datetime format and extract the date
df['timestamp_2'] = pd.to_datetime(df['timestamp'])
df['datestamp'] = df['timestamp_2'].dt.date

# Filter data for the year 2022 only
df = df[(df['timestamp_2'] >= '2022-01-01') & (df['timestamp_2'] < '2023-02-11')]


df_count = df[df['brands'].isin(kw_ls)].groupby(['datestamp', 'brands'])['timestamp_2'].count().reset_index()

# Plot the line graph with 3 lines for each brand
fig, ax = plt.subplots(figsize=(20, 10))

for brand in kw_ls:
    data = df_count[df_count['brands'] == brand].set_index('datestamp')
    data.plot(kind='line', ax=ax, legend=False)

# Add the legend and labels
ax.legend(kw_ls)
ax.set_xlabel('Timeframe')
ax.set_ylabel('Number of Tweets per interval day')
plt.show()



#====correlation
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the data from the csv file
df = pd.read_csv(r'C:\Users\victorlee\Desktop\my_python\JDE\project\result3_tweets.csv')

# Convert the timestamp column to datetime format and extract the date
df['timestamp_2'] = pd.to_datetime(df['timestamp'])
df['datestamp'] = df['timestamp_2'].dt.date

# Filter the data for year 2022
df = df[(df['timestamp_2'] >= '2022-01-01') & (df['timestamp_2'] <= '2023-02-11')]

# Group the data by date and brand,
df_grouped = df.groupby(['timestamp_2', 'brands']).size().reset_index(name='counts')

# Pivot the data so that brands are in columns and dates are in rows
df_pivot = df_grouped.pivot(index='timestamp_2', columns='brands', values='counts')
df_pivotxna = df_pivot.fillna(df_pivot.mean())
# Calculate the correlation between the brands
corr = df_pivotxna.corr()
print(corr)

# Plot the correlation matrix as a heatmap
sns.heatmap(corr, annot=True, cmap='coolwarm')

# Adjust the plot size
plt.gcf().set_size_inches(10, 10)

# Show the plot
plt.show()