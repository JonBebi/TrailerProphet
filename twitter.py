import twint
import nest_asyncio
import pandas as pd
from tqdm import tqdm

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

## Function that takes in a list of tuples of movies and their release date, then scrapes twitter for tweets before the date
## A dictionary is returned where each movie is a key and the values are sentiment stats as well as the dataframe

def what_twitter_thinks(movies):
    sent_dict = {}
    for movie, date in tqdm(movies):
        ## set up the twitter search
        c = twint.Config()
        c.Hide_output = True
        c.Search = movie + ' movie'
        c.Until = date
        c.Pandas = True
        twint.run.Search(c)

        ## create the dataframe
        movie_df = twint.storage.panda.Tweets_df

        ## remove hashtags
        for i in range(len(movie_df)):
            if len(movie_df['hashtags'][i]) > 0:
                movie_df.drop(i,inplace=True)
        ## analyze the tweets
        movie_df['score'] = pd.Series(tweet_analysis(movie_df['tweet']))

        ## add movie and df to sentiment dictionary
        sent_dict[movie] = {'sentiment mean': movie_df['score'].mean(),
                            'sentiment median': movie_df['score'].median(),
                            'sentiment std': movie_df['score'].std(),
                            'full dataframe': movie_df}
    return sent_dict

movies = [('Hustlers','2019-09-12'),('Downton Abbey','2019-09-19'),('Abominable','2019-09-26'),('Ad Astra','2019-09-19')]

big_dict = what_twitter_thinks(movies)
big_dict["Hustlers"]['full dataframe']
