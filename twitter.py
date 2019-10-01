import twint
import nest_asyncio

## Apply nest_asyncio so the event loop in twint doesn't conflict with jupyter notebook
nest_asyncio.apply()

## Instantiate twint
c = twint.Config()

## Collect tweets from "Hustlers" movie and place them in a dataframe
c.Search = "Hustlers movie"
c.Pandas = True
twint.run.Search(c)
hustlers_df = twint.storage.panda.hustlers_df
hustlers_df.head()

## Collect tweets from "Downton Abbey" movie and place them in a dataframe
c.Search = "Downton Abbey"
c.Pandas = True
twint.run.Search(c)
abbey_df = twint.storage.panda.abbey_df

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
def tweet_analysis(series):
    for tweet in series:
        vs = analyzer.polarity_scores(tweet)
        print(f'{tweet}: {str(vs)}')

tweet_analysis(Tweets_df['tweet'])
