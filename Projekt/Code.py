import csv
import pandas as pd
import tweepy
import nltk
from nltk.corpus import stopwords
from nltk.corpus import words
nltk.download('stopwords')
stop = stopwords.words('english')
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import tree
from sklearn.metrics import classification_report

auth = tweepy.AppAuthHandler('', '')
hashtag = "#ETH"
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

#Downloading data from Twitter and moving it over to a CSV file
#csvFile = open('Teaching group.csv', 'a')
#csvWriter = csv.writer(csvFile)
#csvWriter.writerow(['Tweet date', 'Tweet'])
#new_search = hashtag + ' -filter:retweets' + ' -filter:replies'
#for tweet in tweepy.Cursor(api.search, q = new_search, count = 50, lang = 'en', since_id = 0).items(100):
#    csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8')])
#csvFile.close()
with open('Teaching group.csv', 'r', encoding = 'utf-8', newline = '\n') as csvfile:
    csvreader = csv.reader(csvfile)
    date = []
    tweetText = []
    for row in csvreader:
        date.append(row[0])
        tweetText.append(row[1])
pandasTweet = pd.read_csv('Teaching group.csv')
#Analysis
pandasTweet['word_count'] = pandasTweet['Tweet'].apply(lambda x: len(str(x).split(" ")))
pandasTweet[['Tweet','word_count']].head()

pandasTweet['char_count'] = pandasTweet['Tweet'].str.len()
pandasTweet[['Tweet','char_count']].head()

def avg_word(sentence):
  words = sentence.split()
  return (sum(len(word) for word in words)/len(words))

pandasTweet['avg_word'] = pandasTweet['Tweet'].apply(lambda x: avg_word(x))
pandasTweet[['Tweet','avg_word']].head()

pandasTweet['stopwords'] = pandasTweet['Tweet'].apply(lambda x: len([x for x in x.split() if x in stop]))
pandasTweet[['Tweet','stopwords']].head()

pandasTweet['hastags'] = pandasTweet['Tweet'].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))
pandasTweet[['Tweet','hastags']].head()

pandasTweet['numerics'] = pandasTweet['Tweet'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
pandasTweet[['Tweet','numerics']].head()

pandasTweet['upper'] = pandasTweet['Tweet'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
pandasTweet[['Tweet','upper']].head()

#Text cleanup/normalization
pandasTweet['Tweet'] = pandasTweet['Tweet'].replace(r'\\n', ' ', regex=True)
pandasTweet['Tweet'] = pandasTweet['Tweet'].replace(r'\\\w+?', '', regex=True)
pandasTweet['Tweet'] = pandasTweet['Tweet'].replace(r'https:\S+', '', regex=True)
pandasTweet['Tweet']

pandasTweet['Tweet'] = pandasTweet['Tweet'].apply(lambda x: " ".join(x.lower() for x in x.split()))
pandasTweet['Tweet'] = pandasTweet['Tweet'].str.replace('[^\w\s]','')
pandasTweet['Tweet'] = pandasTweet['Tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
pandasTweet['Tweet'] = pandasTweet['Tweet'].apply(lambda x: '' + x[1:])
pandasTweet['Tweet']

freq = pd.Series(' '.join(pandasTweet['Tweet']).split()).value_counts()[:10]
freq

freq = list(freq.index)
pandasTweet['Tweet'] = pandasTweet['Tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
pandasTweet['Tweet'].head()

freq = pd.Series(' '.join(pandasTweet['Tweet']).split()).value_counts()[-20:]
freq

freq = list(freq.index)
pandasTweet['Tweet'] = pandasTweet['Tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
pandasTweet['Tweet']
pandasTweet

pandasTweet['numerics'] = pandasTweet['Tweet'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
pandasTweet[['Tweet','numerics']].head()

pandasTweet['Tweet'] = pandasTweet['Tweet'].str.replace('\d+', '', regex=True)
pandasTweet['Tweet']

pandasTweet['Tweet'] = pandasTweet['Tweet'].apply(lambda x: str(TextBlob(x).correct()))
from textblob import Word
nltk.download('wordnet')
pandasTweet['Tweet'] = pandasTweet['Tweet'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
pandasTweet['Tweet'].head()

TextBlob(pandasTweet['Tweet'][16]).ngrams(2)

tf1 = (pandasTweet['Tweet'][0:99]).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()
tf1.columns = ['words','tf']
tf1

import numpy as np
for i,word in enumerate(tf1['words']):
    tf1.loc[i, 'idf'] = np.log(pandasTweet.shape[0]/(len(pandasTweet[pandasTweet['Tweet'].str.contains(word)])))
tf1

tf1['tfidf'] = tf1['tf'] * tf1['idf']
tf1

pandasTweet['sentiment'] = pandasTweet['Tweet'].apply(lambda x: TextBlob(x).sentiment[0] )
pandasTweet[['Tweet','sentiment']]

predictions = [1,0,1,1,1,1,1,1,1,0,1,0,0,1,0,0,1,1,0,0,0,1,1,1,0,0,1,1,0,1,0,1,0,0,0,0,0,0,0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,1,1,1,1,0,1,0,0,0,1,0,1,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,1,0,0,1,0,1,1]

#Evaluated the tweets and assigned 1 -> i it's positive 0 -> if it's negative/neutral
pandasTweet['Personal evalument'] = predictions
train = []
test = []
pred_train = []
pred_test = []
sent_test = []
train = pandasTweet['Tweet'][0:80]
test = pandasTweet['Tweet'][80:100]
pred_train = pandasTweet['Personal evalument'][0:80]
pred_test = pandasTweet['Personal evalument'][80:100]
sent_test = pandasTweet['sentiment']

#Vectorizing
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(pandasTweet['Tweet'])

xtrain_count =  count_vect.transform(train)
xtest_count =  count_vect.transform(test)
print (xtrain_count)

#Training the model
clf = tree.DecisionTreeClassifier()
clf.fit(xtrain_count, pred_train)

#Prediction
test_results = clf.predict(xtest_count)
test_results

#Predicted outcome:
pred_test

testDataSet = pd.DataFrame(test)

testDataSet['Personal evaluation'] = pred_test
testDataSet['Test results'] = test_results
testDataSet['Sentiment'] = sent_test

testDataSet

print(classification_report(pred_test, test_results))

#Percentage chance of increase in value of ETH
#According to my evaluation based on the tweets
predChance = 7/20 * 100
print(predChance)
#Value should fall
#According to the model
predChanceTest = 13/20 * 100
print(predChanceTest)
#Value should increase
#Time of oldest Tweet within test group:
oldestDate
#2810$
#Value at the time on most recent Tweet
recentDate
#2811

