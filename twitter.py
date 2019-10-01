import twint
import nest_asyncio
import pandas as pd

## Apply nest_asyncio so the event loop in twint doesn't conflict with jupyter notebook
nest_asyncio.apply()

## Analyze the sentiment of all the tweets
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

## Function takes in the pandas series that holds all of the tweets and hold them in a list
def tweet_analysis(series):
    score_list = []
    for tweet in series:
        vs = analyzer.polarity_scores(tweet)
        score_list.append(vs['compound'])
    return score_list

## Function that takes in a movie and a release date, then scrapes twitter for 3000 tweets before the date
## analyzes all the tweets and returns the mean

def what_twitter_thinks(movie, date):
    ## set up the twitter search
    c = twint.Config()
    c.Search = movie
    c.Until = date
    c.Pandas = True
    twint.run.Search(c)

    ## create the dataframe
    movie_df = twint.storage.panda.Tweets_df
    movie_df['score'] = pd.Series(tweet_analysis(movie_df['tweet']))

    return movie_df['score'].mean()

## Analyze Hustlers movie
hustlers = "Hustlers movie"
hustlers_date = '2019-09-12'

## Analyze Downton Abbey
downton_abbey = "Downton Abbey movie"
da_date = "2019-09-19"

## Analyze Abominable
abominable = "Abominable movie"
abominable_date = "2019-09-26"
