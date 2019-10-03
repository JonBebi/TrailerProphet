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
        c.Videos = True
        c.Until = date
        c.Limit = 2000
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

## Grab movie titles and release dates
movies_and_dates = [
                  ('Crawl','2019-07-10'),
                  ('Stuber','2019-07-10'),
                  ('The Art of Self-Defense','2019-07-10'),
                  ('Bethany Hamilton: Unstoppable','2019-07-10'),
                  ('The Farewell','2019-07-10'),
                  ('Sword of Trust','2019-07-11'),
                  ('The Lion King','2019-07-17'),
                  ('Luz','2019-07-17'),
                  ('The Great Hack','2019-07-17'),
                  ('Once Upon a Time in Hollywood','2019-07-24'),
                  ('The Ground Beneath My Feet','2019-07-24'),
                  ('Honeyland ','2019-07-24'),
                  ('Mike Wallace Is Here ','2019-07-24'),
                  ('The Mountain ','2019-07-24'),
                  ('Fast & Furious Presents: Hobbs & Shaw','2019-08-01'),
                  ('Love, Antosha','2019-08-01'),
                  ('Luce','2019-08-01'),
                  ('The Nightingale','2019-08-01'),
                  ('Otherhood','2019-08-01'),
                  ('Them That Follow','2019-08-01'),
                  ('The Art of Racing in the Rain','2019-08-7'),
                  ('Dora and the Lost City of Gold','2019-08-7'),
                  ('The Kitchen','2019-08-7'),
                  ('Scary Stories to Tell in the Dark','2019-08-7'),
                  ('After the Wedding','2019-08-7'),
                  ('Light of My Life','2019-08-8'),
                  ('The Angry Birds Movie 2','2019-08-7'), ('Los Reyes','2019-08-6'),
                  ('47 Meters Down: Uncaged','2019-08-14'),
                  ('Blinded By the Light','2019-08-14'),
                  ('Good Boys','2019-08-14'),
                  ("Where'd You Go, Bernadette",'2019-08-14'),
                  ('Angel Has Fallen','2019-08-21'),
                  ('Overcomer','2019-08-21'),
                  ('Brittany Runs a Marathon','2019-08-21'),
                  ('Fiddler: A Miracle of Miracles','2019-08-22'),
                  ('Give Me Liberty','2019-08-23'),
                  ('It Chapter Two','2019-09-5'), ('Blink of an Eye','2019-09-5'), ('Ms. Purple','2019-09-5'),
                  ('The Goldfinch','2019-09-12'),
                  ('Hustlers','2019-09-12'),
                  ('Ad Astra','2019-09-19'),
                  ('Downton Abbey','2019-09-19'),
                  ('Rambo: Last Blood','2019-09-19'),
                  ('Abominable','2019-09-26'),
                  ('Judy','2019-09-26'),
                  ('Prey','2019-09-26'),
                  ('Spiderman: Far from Home','2019-07-01'),
                  ('Toy Story 4','2019-06-23')]

## Create a dictionary of movies with their sentiment scores and individual dataframe
the_dictionary = what_twitter_thinks(movies_and_dates)

### Save/open pickled dictionary
import pickle
# 'sentiment mean': 0.031474773139745935, 'sentiment median': 0.0, 'sentiment std': 0.48650541149816
the_dictionary = pickle.load(open('/home/xristsos/flatiron/projects/trailer_prophet/twitter50sentiment.pickle','rb'))
# the_dictionary['Hustlers']['sentiment mean']
# pickle.dump(the_dictionary,open('twitter50sentiment_just_videos.pickle','wb'))


####### Prints the scores for each movie ######
### Not very useful right now but it might be ###
def scores(dictionary):
    scores = ['sentiment mean','sentiment std','sentiment median']
    for movie, keys in dictionary.items():
        print(f'{movie}:\n -----------------------')
        for score, value in keys.items():
            if score in scores:
                print(f'{score}: {value}')

#### Function that takes in the dictionary and merges all the dataframes from each movie into one
def merge_dataframes(dictionary):
    ### Starts with the first dataframe of the first movie
    starter_df = pd.DataFrame(list(dictionary.items())[0][1]['full dataframe'])
    starter_df = starter_df.reset_index(drop=True)
    ### Loops through the dictionies to isolate the movie's dataframe
    for movie, df_keys in tqdm(dictionary.items()):
        for key, df in df_keys.items():
            if key == 'full dataframe':
                starter_df = pd.concat([starter_df,df])
            else:
                continue
    return starter_df

## scores(the_dictionary)
df = merge_dataframes(the_dictionary)

## Get rid of all tweets with 0 sentiment
df = df[df['score']!=0]

## Keep relevent columns
df = df[['score','nlikes','nreplies','nretweets','retweet','retweet_date','search','tweet','reply_to']]

## dataframe cleaning
##### Gets rid of the word 'movie' in 'search' column and keeps the title of the movie in a new column called 'movie'
df['movie'] = [x.replace(' movie','') for x in df['search']]
##### Drop search column since we don't need it anymore
df.drop(columns='search',inplace=True)
##### There are chunks of sentiment scores missing for each movie so it was replaced with the mean
df.fillna(df['score'].mean(),inplace=True)

## Add RT scores to dataframe
rt_dict = {'Abominable': 1,
           'Blink of an Eye': 1,
           'Downton Abbey': 1,
           'Hustlers': 1,
           'It Chapter Two': 1,
           'Ad Astra': 1,
           'Rambo: Last Blood': 0,
           'Judy': 1,
           'Good Boys': 1,
           'The Lion King': 0,
           'Angel Has Fallen': 0,
           'Fast & Furious Presents: Hobbs & Shaw': 1,
           'Overcomer': 0,
           'The Peanut Butter Falcon': 1,
           'Scary Stories to Tell in the Dark': 1,
           'Dora and the Lost City of Gold': 1,
           'Brittany Runs a Marathon': 1,
           'Once Upon a Time in Hollywood': 1,
           'Linda Ronstadt: The Sound of My Voice': 1,
           'The Angry Birds Movie 2': 1,
           'Crawl': 1,
           'Stuber': 0,
           'The Art of Self-Defense': 1,
           'Bethany Hamilton: Unstoppable': 1,
           'The Farewell': 1,
           'Sword of Trust': 1,
           'Luz': 1,
           'The Great Hack':1,
           'The Ground Beneath My Feet':1,
           'Honeyland': 1,
           'Mike Wallace Is Here': 1,
           'The Mountain': 1,
           'Love, Antosha': 1,
           'Luce': 1,
           'The Nightingale': 1,
           'Otherhood': 0,
           'Them That Follow': 1,
           'The Art of Racing in the Rain': 0,
           'The Kitchen': 1,
           'After the Wedding': 0,
           'Light of My Life': 1,
           '47 Meters Down: Uncaged': 0,
           'Blinded By the Light': 1,
           "Where'd You Go, Bernadette": 0,
           'Fiddler: A Miracle of Miracles': 1,
           'Give Me Liberty': 1,
           'The Goldfinch': 0,
           'Toy Story 4': 1,
           'Los Reyes': 1,
           'Ms. Purple': 1,
           'Prey': 0,
           'Spiderman: Far from Home': 1}

### Create a list of movies
movies = df['movie'].unique()

### Function that takes in the list of movies and the dataframe and returns a new dictionary
### New dictionary is a summary of each movie's twitter activity
def store_movie_stats(movies,data):
    clean_dict = {}
    for movie in movies:
        movie_df = data[data['movie']==movie]
        avg_sentiment = movie_df['score'].mean()
        total_likes = movie_df['nlikes'].sum()
        total_retweets = movie_df['nretweets'].sum()
        total_replies = movie_df['nreplies'].sum()
        total_tweets = len(movie_df)

        ## Update clean dictionary
        clean_dict[movie] = {'avg_sentiment': avg_sentiment,
                             'total_likes': total_likes,
                             'total_retweets': total_retweets,
                             'total_replies': total_replies,
                             'total tweets': total_tweets}
    return clean_dict

### Create a new dataframe from the new dictionary
df_movies = pd.DataFrame(store_movie_stats(movies,df))

### Thew new dataframe has the movies as the columns
### The ".T" puts the rows into the columns and the columns into the rows
df_movies = df_movies.T

### Reset the index so the movie titles have their own column
df_movies.reset_index(inplace=True)
df_movies.rename(columns={'index': 'movie'},inplace=True)

### Strip any white space from the movie titles, just in case
df_movies['movie'] = df_movies['movie'].apply(lambda x: x.strip())

### Create an empty list for the Rotten Tomatoes scores
### Loop through dataframe to fill the empty list with the appropriate rt score
rt_scores = []
for i in range(len(df_movies)):
    rt_scores.append(rt_dict[df_movies.loc[i]['movie']])
df_movies['freshness'] = rt_scores

## Feature idea: tokenized words -- do key words help predict rt score
