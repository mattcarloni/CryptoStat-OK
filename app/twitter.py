from flask import (
    Blueprint,
    render_template,
    request,
)
from .extentions import mongo
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import tweepy
from textblob import TextBlob
import re

twitter = Blueprint('twitter', __name__)

@twitter.route("/sentiment")
def sentiment():
    return render_template("sentiment.html")

@twitter.route('/get_sentiment', methods=['POST','GET'])
def get_sentiment():
      
    if request.method == 'POST':

        #Set the twiter api credentials
        consumerKey = "FBqzTcadgCRFPI43737Qf1UpJ"
        consumerSecret = "GDBgg6TEU38xoJA7CBXeSKPqo18KEEjWYxqTu3jPYglQGqHylG"
        accessToken = "928261732539052032-lCMV974DMEgdjYYMLKMkUVaIzcLzRfL"
        accessTokenSecret = "knlt9Nw4MdFgiopdARBcNbHq33oZZUhRoktU5fRfL9EPn"

        #Create the authentication object, set the tokens and the API access
        authenticate = tweepy.OAuthHandler(consumerKey, consumerSecret)
        authenticate.set_access_token(accessToken, accessTokenSecret)
        api = tweepy.API(authenticate, wait_on_rate_limit=True)

        global b
        b = request.form['cryptotwt']

        #Gather 100 tweets and filter out any retweets and store them in a variable
        search_term = '#' + b + '-filter:retweets'
        tweets = tweepy.Cursor(api.search_tweets, q=search_term, lang='en', tweet_mode='extended').items(100)
        all_tweets = [tweet.full_text for tweet in tweets]

        #Create a dataframe to store the tweets with a column called 'Tweets'
        df = pd.DataFrame(all_tweets, columns=['Tweets'])
        
        #Create a function to clean the tweets
        def cleanTwt(twt):
            twt = re.sub('#[A-Za-z0-9]+', '', twt) #Removes any strings with a '#'
            twt = re.sub('\\n', '', twt) #Removes the '\n' string
            twt = re.sub('https?:\/\/\S+', '', twt) #Removes any hyperlinks
            twt = re.sub('@[A-za-z0-9]+', '', twt) #Removes any mentions
            return twt

        df['Cleaned_Tweets'] = df['Tweets'].apply(cleanTwt)

        #Create a function to get the subjectivity
        def getSubjectivity(twt):
            return TextBlob(twt).sentiment.subjectivity
        #Create a function to get the polarity
        def getPolarity(twt):
            return TextBlob(twt).sentiment.polarity

        df['Subjectivity'] = df['Cleaned_Tweets'].apply(getSubjectivity)
        df['Polarity'] = df['Cleaned_Tweets'].apply(getPolarity)

        #Create a function to get the sentiment text
        def getSentiment(score):
            if score < 0:
                return 'Negative'
            elif score == 0:
                return 'Neutral'
            else:
                return 'Positive'

        df['Sentiment'] = df['Polarity'].apply(getSentiment)

        #Create a scatter plot to show the subjectivity and the polarity
        plt.figure(figsize=(8, 6), dpi=105)
        for i in range(0, df.shape[0]):
            plt.style.use('fivethirtyeight')
            plt.scatter(df['Polarity'][i], df['Subjectivity'][i], color='Purple')
            plt.title('Sentiment Analysis Scatter Plot')
            plt.xlabel('Polarity')
            plt.ylabel('Subjectivity (Fact -> Opinion)')
            plt.tight_layout()
            plt.savefig('app/static/media/my_plot_2.png')

        #Create a bar chart to show the count of Positive, Neutral, and Negative Sentiment
        plt.style.use('fivethirtyeight')
        plt.figure(figsize=(8, 6), dpi=105)
        df['Sentiment'].value_counts().plot(kind='bar')
        plt.title('Sentiment Analysis Bar Plot')
        plt.xlabel('Sentiment')
        plt.ylabel('Number of tweets')
        plt.tight_layout()
        plt.savefig('app/static/media/my_plot_3.png')
            
    return render_template('sentiment.html', get_sentiment = True, plot_url2 = 'static/media/my_plot_2.png', plot_url3 = 'static/media/my_plot_3.png')
