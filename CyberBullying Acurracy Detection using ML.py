#!/usr/bin/env python
# coding: utf-8

# In[17]:


import re 
import tweepy 
from tweepy import OAuthHandler 
from textblob import TextBlob 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 
# import sqlalchemy as sa
# from sqlalchemy import create_engine
import pandas as pd 
import sqlite3 
import warnings 
import numpy as np 
# for data manipulation 
import matplotlib.pyplot as plt
import nltk  # for text manipulation 
import string # for text manipulation 
from nltk.stem.porter import * 


from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression 
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer





pd.set_option("display.max_colwidth", 200) 
warnings.filterwarnings("ignore") #ignore warnings


# In[18]:


TextBlob=0
Vader=0
TwiterApiEnable=0
FileName="training.1600000.processed.noemoticon.csv"


# In[36]:


def initTwitter():
    # keys and tokens from twitter developer app  
#     replaced by the word key

    ConsumerKey = 'key'
    ConsumerSecret = 'key'
    AccessToken = 'key'
    AccessTokenSecret = 'key'

    #  authentication twitter using API keys 
    api_object=''
    try: 

        authenticate = OAuthHandler(ConsumerKey, ConsumerSecret) 

        authenticate.set_access_token(AccessToken, AccessTokenSecret) 

        api_object = tweepy.API(authenticate) 
    except Exception as e: 
        print("Error: Authentication Failed",e) 
    return api_object


# In[20]:


def ParseTweet(tweet): 
    ''' 
    function that make tweets cleaner by removing links, special characters 
    using simple regex statements. 
    '''
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) | (\w+:\/\/\S+)", " ", tweet).split()) 


# In[21]:


def GeTweetOpinionTextblob(tweet): 
    ''' 
    classifies opinions of passed tweet 
    using textblob's sentiment method 
    '''
    tweet_analysis = TextBlob(self.ParseTweet(tweet)) 
    # set sentiment 
    if tweet_analysis.sentiment.polarity > 0: 
        return 4
    elif tweet_analysis.sentiment.polarity == 0: 
        return 2
    else: 
        return 0


# In[22]:


def GeTweetOpinionVader(tweet): 
    analyser = SentimentIntensityAnalyzer() 
    # polarity_scores method of SentimentIntensityAnalyzer 
    analyser_data = analyser.polarity_scores(tweet) 

    if analyser_data['compound'] >= 0.05 : 
        return 4 

    elif analyser_data['compound'] <= - 0.05 : 
        return 0 
    else : 
        return 2 


# In[50]:


def GeTweets(api_object,query, count ): 
    ''' 
    Main function to fetch tweets and parse them. 
    '''
    # empty list to store parsed tweets 
    list_tweets = [] 
    list_value=[]


    try: 
        # call twitter api to fetch tweets 
#          here query can be hash tags and we can also pass an argument for geocode which decides location. 
#  geocode="latitude,longitude,radius" in this format. 

#         tweets_api = api_object.search(q = query, count = count,gecode='37.781157,-122.398720,1mi')
        tweets_api = api_object.search(q = query, count = count)

    # print(fetched_tweets) 

        # parsing tweets one by one 
        for tweet in tweets_api: 
            # empty dictionary to store required params of a tweet 
            opinion=0
            if TextBlob==1:
                opinion= GeTweetOpinionTextblob(tweet.text) 
            #  Vader sentimental Results 
            else:
                opinion = GeTweetOpinionVader(tweet.text) 
            if tweet.retweet_count > 0: 
                # if retweets 
                if tweet.text not in list_tweets: 
                    list_tweets.append(tweet.text)
                    list_value.append(opinion)
            else: 
                list_tweets.append(tweet.text) 
                list_value.append(opinion)
        dict_df={'target':list_value,'Tweet':list_tweets}
        df=pd.DataFrame(dict_df)
        return df
    except tweepy.TweepError as e: 
        # print error (if any) 
        print("Error : " + str(e)) 


# In[43]:


def getTweetsFromApi(api_object,keywords,count):
    tweets=GeTweets(api_object,keywords, count)
    return dftweets


# In[ ]:





# In[25]:


def getTweetsFromDataBase():
    data = pd.read_csv(FileName,encoding='latin-1')
    columnNames = ["target", "ids", "date", "flag", "user", "TweetText"]
    data.columns =columnNames 
    data.drop(['ids','date','flag','user'],axis = 1,inplace = True)
    return data 
    


# In[26]:


def PostaggingStemming(data):
#   removing Stopwords (Pos Tagging)
    stopwords=nltk.corpus.stopwords.words('english')
    data['TweetsCleaned'] = data['TweetsCleaned'].apply(lambda text : stopwordsRemove(text.lower(),stopwords))
#   Tokenization 
    data['TweetsCleaned'] = data['TweetsCleaned'].apply(lambda x: x.split())
#   Applying Steeming 
    stemmer = PorterStemmer() 
    data['TweetsCleaned'] = data['TweetsCleaned'].apply(lambda x: [stemmer.stem(i) for i in x])
#   join the tweets after doing stemming .
    data['TweetsCleaned'] = data['TweetsCleaned'].apply(lambda x: ' '.join([w for w in x]))
#   Removing small words length >3 will be included (is are etc removed)
    data['TweetsCleaned'] = data['TweetsCleaned'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
    
    return data 
    




    


# In[27]:


def DataCleaning(data):
#   Making the database of positive and negative comments  
#  these can be changed right now taking 26000 only.
    positive = data[data.target==4].iloc[:25000,:]
    negative = data[data.target==0].iloc[:1000,:]
    data = pd.concat([positive,negative],axis = 0)
#   cleaning the tweets :
    data['TweetsCleaned']=data['TweetText'].apply(ParseTweet)
# 
    
    return data 


# In[ ]:





# In[28]:


# def Datavisualization():
def stopwordsRemove(text,stopwords):
    clean_text=' '.join([word for word in text.split() if word not in stopwords])
    return clean_text
    


# In[29]:


def model(data):
    X_train,X_test,y_train,y_test= Training_and_predict(data)
#      xgBoost 
    xgbc = XGBClassifier(max_depth=6, n_estimators=1000, nthread= 3)
    xgbc.fit(X_train,y_train)
    prediction_xgb = xgbc.predict(X_test)
    print('xgb boost accuracy : ',accuracy_score(prediction_xgb,y_test))
#   Random Forest
    rf = RandomForestClassifier(n_estimators=1000, random_state=42)
    rf.fit(X_train,y_train)
    prediction_rf = rf.predict(X_test)
    print('random forest  accuracy :',accuracy_score(prediction_rf,y_test))

#     logistic regression
    lr = LogisticRegression()
    lr.fit(X_train,y_train)
    prediction_lr = lr.predict(X_test)
    print('logistic accuracy : ',accuracy_score(prediction_lr,y_test))

#  SVM 

    svc = svm.SVC()
    svc.fit(X_train,y_train)
    prediction_svc = svc.predict(X_test)
    print('Svm accuracy : ',accuracy_score(prediction_svc,y_test))

    
    


# In[30]:


def Training_and_predict(data):
    count_vectorizer = CountVectorizer(stop_words='english') 
    cv = count_vectorizer.fit_transform(data['TweetsCleaned'])
#      Training and test Data 
    X_train,X_test,y_train,y_test = train_test_split(cv,data['target'] , test_size=.2,stratify=data['target'], random_state=42)
    return X_train,X_test,y_train,y_test


# In[ ]:





# In[31]:


# program Stars 
# Use this section to print result with data base 
# Data 
if TwiterApiEnable==1:
    api_object=initTwitter()
    keywords='i hate negros'
    counts=1000
    dftweets=getTweetsFromApi(api_object,keywords,counts)
else:
    dftweets=getTweetsFromDataBase()
    
# Data cleaning 
dftweets=DataCleaning(dftweets)
#  Pos Tagging 
dftweets=PostaggingStemming(dftweets)
# model accuracy 
model(dftweets)


# In[ ]:


# program Stars 
# Use this section to print result with Tweets Api  
# Data 
TwiterApiEnable=1
if TwiterApiEnable==1:
    api_object=initTwitter()
    keywords='Donald Trump'
    counts=1000
    dftweets=getTweetsFromApi(api_object,keywords,counts)
else:
    dftweets=getTweetsFromDataBase()
    
# Data cleaning 
dftweets=DataCleaning(dftweets)
#  Pos Tagging 
dftweets=PostaggingStemming(dftweets)
# model accuracy 
model(dftweets)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




