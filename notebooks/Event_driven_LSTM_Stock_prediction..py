#!/usr/bin/env python
# coding: utf-8

# # Part 0. Import Module

# In[180]:


import sys, csv, json
import requests
from newsapi.articles import Articles
from newsapi.sources import Sources
import numpy as np
import csv, json
import pandas as pd
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import unicodedata
import math
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops


# # Part 1. Collect NYTimes data
#
# Newsapi documentation
#
# https://github.com/SlapBot/newsapi
#
# https://github.com/llSourcell/Stock_Market_Prediction/blob/master/Collecting%20NYTimes%20Data.py

# In[16]:


key = '96af62a035db45bda517a9ca62a25ac3'


# In[18]:


a = Articles(API_KEY=key)
s = Sources(API_KEY=key)


# Creating 2 exception
#
# 1 for API
#
# 1 for Timeframe for the news

# In[19]:


class APIKeyException(Exception):
    def __init__(self, message):
        self.message = message

class InvalidQueryException(Exception):
    def __init__(self, message):
        self.message = message


# Initializes the ArchiveAPI class to downlaod data to json file
#
# Raises an exception if no API key is given.
#
# param key: New York Times API Key

# In[21]:


class ArchiveAPI(object):
    def __init__(self, key=None):
        self.key = key
        self.root = 'http://api.nytimes.com/svc/archive/v1/{}/{}.json?api-key={}'
        if not self.key:
            nyt_dev_page = 'http://developer.nytimes.com/docs/reference/keys'
            exception_str = 'Warning: API Key required. Please visit {}'
            raise NoAPIKeyException(exception_str.format(nyt_dev_page))

    def query(self, year=None, month=None, key=None,):
        """
        Calls the archive API and returns the results as a dictionary.
        :param key: Defaults to the API key used to initialize the ArchiveAPI class.
        """
        if not key:
            key = self.key

        if (year < 1882) or not (0 < month < 13):
            # currently the Archive API only supports year >= 1882
            exception_str = 'Invalid query: See http://developer.nytimes.com/archive_api.json'
            raise InvalidQueryException(exception_str)

        url = self.root.format(year, month, key)
        r = requests.get(url)
        return r.json()


api = ArchiveAPI('0ba6dc04a8cb44e0a890c00df88c393a')


years = [2016, 2015, 2014, 2013, 2012, 2011, 2010, 2009, 2008, 2007]
months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

''' for year in years:
    for month in months:
        mydict = api.query(year, month)
        file_str = './stock_rnn_data/' + str(year) + '-' + '{:02}'.format(month) + '.json'
        with open(file_str, 'w') as fout:
            try:
                json.dump(mydict, fout)
            except:
                pass
        fout.close()
'''

# # Part 2. Preparing Stock Price
#
# Currently using a csv file, will use Quandl to download more and live data soon

# In[68]:

with open('./data/DJIA_indices_data.csv', 'r',encoding="utf-8") as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    # Converting the csv file reader to a lists
    data_list = list(spamreader)


# Separating header from the data

# In[69]:


header = data_list[0]
data_list = data_list[1:]

data_list = np.asarray(data_list)


# In[70]:


print(data_list)


# Selecting date and close value and adj close for each day
#
# volume will be added in the next model

# In[71]:


selected_data = data_list[:, [0, 4, 6]]


# Convert it into dataframe
#
# index = date

# In[72]:


df = pd.DataFrame(data=selected_data[0:,1:],
             index=selected_data[0:,0],
                                columns=['close', 'adj close'],
                                        dtype='float64')


# In[73]:


print (df.tail())


# Interpolating data

# In[99]:


df1 = df
idx = pd.date_range('12-29-2006', '12-31-2016')
df1.index = pd.DatetimeIndex(df1.index)
df1 = df1.reindex(idx, fill_value=np.NaN)
# df1.count() # gives 2518 count
interpolated_df = df1.interpolate() # Fill in the gap
interpolated_df.count() # gives 3651 count


# In[100]:


print (df1.head(25))


# In[101]:


# Removing extra date rows added in data for calculating interpolation
interpolated_df = interpolated_df[3:]


# In[102]:


print (interpolated_df.head())


# # Part 3. Merging NYTimes data

# Function to parse and convert date format
#
# Try 2 formats for date or raise error

# In[103]:


date_format = ["%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S+%f"]

def try_parsing_date(text):
    for fmt in date_format:
        try:
            return datetime.strptime(text, fmt).strftime('%Y-%m-%d')
        except ValueError:
            pass
    raise ValueError('no valid date format found')


# In[104]:


years = [2016, 2015, 2014, 2013, 2012, 2011, 2010, 2009, 2008, 2007]
months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
dict_keys = ['pub_date', 'headline'] #, 'lead_paragraph']
articles_dict = dict.fromkeys(dict_keys)


# Filtering to read only the following news

# In[105]:


# Filtering list for type_of_material
type_of_material_list = ['blog', 'brief', 'news', 'editorial', 'op-ed', 'list','analysis']
# Filtering list for section_name
section_name_list = ['business', 'national', 'world', 'u.s.' , 'politics', 'opinion', 'tech', 'science',  'health']
news_desk_list = ['business', 'national', 'world', 'u.s.' , 'politics', 'opinion', 'tech', 'science',  'health', 'foreign']


# In[106]:


current_date = '2016-10-01'
from datetime import datetime

current_article_str = ''


# Adding article column to dataframe

# In[107]:


interpolated_df["articles"] = ''
count_articles_filtered = 0
count_total_articles = 0
count_main_not_exist = 0
count_unicode_error = 0
count_attribute_error = 0


# In[108]:

'''
for year in years:  # search for every month
    for month in months:
        file_str = './stock_rnn_data/' + str(year) + '-' + '{:02}'.format(month) + '.json'
        with open(file_str) as data_file:
            NYTimes_data = json.load(data_file)
        count_total_articles = count_total_articles + len(NYTimes_data["response"]["docs"][:]) #add article number
        for i in range(len(NYTimes_data["response"]["docs"][:])): # search in every docs for type of material or section = in the list
            try:
                if any(substring in NYTimes_data["response"]["docs"][:][i]['type_of_material'].lower() for substring in type_of_material_list):
                    if any(substring in NYTimes_data["response"]["docs"][:][i]['section_name'].lower() for substring in section_name_list):
                        #count += 1
                        count_articles_filtered += 1
                        #print 'i: ' + str(i) dick_key = ['pub_date', 'headline']
                        articles_dict = { your_key: NYTimes_data["response"]["docs"][:][i][your_key] for your_key in dict_keys }
                        articles_dict['headline'] = articles_dict['headline']['main'] # Selecting just 'main' from headline
                        #articles_dict['headline'] = articles_dict['lead_paragraph'] # Selecting lead_paragraph
                        date = try_parsing_date(articles_dict['pub_date'])
                        #print 'article_dict: ' + articles_dict['headline']
                        # putting same day article str into one str
                        if date == current_date:
                            current_article_str = current_article_str + '. ' + articles_dict['headline']
                        else:
                            interpolated_df.set_value(current_date, 'articles', interpolated_df.loc[current_date, 'articles'] + '. ' + current_article_str)
                            current_date = date
                            #interpolated_df.set_value(date, 'articles', current_article_str)
                            #print str(date) + current_article_str
                            current_article_str = articles_dict['headline']
                        # For last condition in a year
                        if (date == current_date) and (i == len(NYTimes_data["response"]["docs"][:]) - 1):
                            interpolated_df.set_value(date, 'articles', current_article_str)

             #Exception for section_name or type_of_material absent
            except AttributeError:
                #print 'attribute error'
                #print NYTimes_data["response"]["docs"][:][i]
                count_attribute_error += 1
                # If article matches news_desk_list if none section_name found
                try:
                    if any(substring in NYTimes_data["response"]["docs"][:][i]['news_desk'].lower() for substring in news_desk_list):
                            #count += 1
                            count_articles_filtered += 1
                            #print 'i: ' + str(i)
                            articles_dict = { your_key: NYTimes_data["response"]["docs"][:][i][your_key] for your_key in dict_keys }
                            articles_dict['headline'] = articles_dict['headline']['main'] # Selecting just 'main' from headline
                            #articles_dict['headline'] = articles_dict['lead_paragraph'] # Selecting lead_paragraph
                            date = try_parsing_date(articles_dict['pub_date'])
                            #print 'article_dict: ' + articles_dict['headline']
                            if date == current_date:
                                current_article_str = current_article_str + '. ' + articles_dict['headline']
                            else:
                                interpolated_df.set_value(current_date, 'articles', interpolated_df.loc[current_date, 'articles'] + '. ' + current_article_str)
                                current_date = date
                                #interpolated_df.set_value(date, 'articles', current_article_str)
                                #print str(date) + current_article_str
                                current_article_str = articles_dict['headline']
                            # For last condition in a year
                            if (date == current_date) and (i == len(NYTimes_data["response"]["docs"][:]) - 1):
                                interpolated_df.set_value(date, 'articles', current_article_str)

                except AttributeError:
                    pass
                pass
            except KeyError:
                print ('key error')
                #print NYTimes_data["response"]["docs"][:][i]
                count_main_not_exist += 1
                pass
            except TypeError:
                print ("type error")
                #print NYTimes_data["response"]["docs"][:][i]
                count_main_not_exist += 1
                pass


# In[110]:


print (count_articles_filtered)
print (count_total_articles)
print (count_main_not_exist)
print (count_unicode_error)


# In[112]:


## Putting all articles if no section_name or news_desk not found
for date, row in interpolated_df.T.iteritems():
    if len(interpolated_df.loc[date, 'articles']) <= 400:
        #print interpolated_df.loc[date, 'articles']
        #print date
        month = date.month
        year = date.year
        file_str = './stock_rnn_data/' + str(year) + '-' + '{:02}'.format(month) + '.json'
        with open(file_str) as data_file:
            NYTimes_data = json.load(data_file)
        count_total_articles = count_total_articles + len(NYTimes_data["response"]["docs"][:])
        interpolated_df.set_value(date.strftime('%Y-%m-%d'), 'articles', '')
        for i in range(len(NYTimes_data["response"]["docs"][:])):
            try:

                articles_dict = { your_key: NYTimes_data["response"]["docs"][:][i][your_key] for your_key in dict_keys }
                articles_dict['headline'] = articles_dict['headline']['main'] # Selecting just 'main' from headline
                #articles_dict['headline'] = articles_dict['lead_paragraph'] # Selecting lead_paragraph
                pub_date = try_parsing_date(articles_dict['pub_date'])
                #print 'article_dict: ' + articles_dict['headline']
                if date.strftime('%Y-%m-%d') == pub_date:
                    interpolated_df.set_value(pub_date, 'articles', interpolated_df.loc[pub_date, 'articles'] + '. ' + articles_dict['headline'])

            except KeyError:
                print ('key error')
                #print NYTimes_data["response"]["docs"][:][i]
                #count_main_not_exist += 1
                pass
            except TypeError:
                print ("type error")
                #print NYTimes_data["response"]["docs"][:][i]
                #count_main_not_exist += 1
                pass


# In[113]:


# Saving the data as pickle file
interpolated_df.to_pickle('./stock_rnn_data/pickled_ten_year_filtered_lead_para.pkl')


# Save pandas frame in csv form
interpolated_df.to_csv('./stock_rnn_data/sample_interpolated_df_10_years_filtered_lead_para.csv',
                       sep='\t', encoding='utf-8')
'''

print("Here")
# Reading the data as pickle file
interpolated_df = pd.read_pickle("./stock_rnn_data/pickled_ten_year_filtered_lead_para.pkl")
dataframe_read = pd.read_pickle('./stock_rnn_data/pickled_ten_year_filtered_lead_para.pkl')



# # Part 4. Deep Neural Network
#
# start here

# In[121]:


df_stocks = pd.read_pickle('./stock_rnn_data/pickled_ten_year_filtered_lead_para.pkl')


# In[122]:


print (df_stocks.head())


# In[123]:


df_stocks['prices'] = df_stocks['adj close'].apply(np.int64)


# In[124]:


# selecting the prices and articles
df_stocks = df_stocks[['prices', 'articles']]


# In[135]:


df_stocks.head()


# Removing . or - from the beginning

# In[127]:


df_stocks['articles'] = df_stocks['articles'].map(lambda x: x.lstrip('.-'))


# In[134]:


df_stocks.head()


# In[129]:


df = df_stocks[['prices']].copy()


# In[130]:


df.head()


# Adding new columns to the data frame

# In[133]:


df["compound"] = ''
df["neg"] = ''
df["neu"] = ''
df["pos"] = ''


# In[136]:


df.head()


# In[144]:


df_stocks.T


# In[145]:


#nltk.download()


# unicodedata.normalize  = Return the normal form form for the Unicode string unistr.
#

# In[338]:


sid = SentimentIntensityAnalyzer()
for date, row in df_stocks.T.iteritems():
    try:
        sentence = unicodedata.normalize('NFKD', df_stocks.loc[date, 'articles'])
        ss = sid.polarity_scores(sentence)
        df.set_value(date, 'compound', ss['compound'])
        df.set_value(date, 'neg', ss['neg'])
        df.set_value(date, 'neu', ss['neu'])
        df.set_value(date, 'pos', ss['pos'])
    except TypeError:
        print (df_stocks.loc[date, 'articles'])
        print (date)


# In[339]:


df.head()


# Gonna use numpy normalize to replace this formula soon

# In[340]:


datasetNorm = (df - df.mean()) / (df.max() - df.min())
datasetNorm.reset_index(inplace=True)
del datasetNorm['index']
datasetNorm['next_prices'] = datasetNorm['prices'].shift(-1)
datasetNorm.head(5)


# Hyperparameters

# In[366]:


num_epochs = 50

batch_size = 1

total_series_length = len(datasetNorm.index)

truncated_backprop_length = 3 #The size of the sequence

state_size = 12 #The number of neurons

num_features = 4
num_classes = 1 #[1,0]

num_batches = total_series_length//batch_size//truncated_backprop_length

min_test_size = 100

print('The total series length is: %d' %total_series_length)
print('The current configuration gives us %d batches of %d observations each one looking %d steps in the past'
      %(num_batches,batch_size,truncated_backprop_length))


# Train-Test split

# In[367]:


datasetTrain = datasetNorm[datasetNorm.index < num_batches*batch_size*truncated_backprop_length]


for i in range(min_test_size,len(datasetNorm.index)):

    if(i % truncated_backprop_length*batch_size == 0):
        test_first_idx = len(datasetNorm.index)-i
        break

datasetTest =  datasetNorm[datasetNorm.index >= test_first_idx]


# In[368]:


xTrain = datasetTrain[['prices','neu','neg','pos']].as_matrix()
yTrain = datasetTrain['next_prices'].as_matrix()


# In[369]:


xTrain.shape


# In[370]:


xTest = datasetTest[['prices','neu','neg','pos']].as_matrix()
yTest = datasetTest['next_prices'].as_matrix()


# In[371]:


yTest.shape


# Visualize starting price data

# In[372]:


plt.figure(figsize=(25,5))
plt.plot(xTrain[:,0])
plt.title('Train (' +str(len(xTrain))+' data points)')
plt.show()
plt.figure(figsize=(10,3))
plt.plot(xTest[:,0])
plt.title('Test (' +str(len(xTest))+' data points)')
plt.show()


# Placeholders

# In[384]:


tf.reset_default_graph()


# In[385]:


batchX_placeholder = tf.placeholder(dtype=tf.float32,shape=[None,truncated_backprop_length,num_features],name='data_ph')
batchY_placeholder = tf.placeholder(dtype=tf.float32,shape=[None,truncated_backprop_length,num_classes],name='target_ph')


# Weights and biases
#
# Because is a 3 layer net:
#
# Input
#
# Hidden Recurrent layer
#
# Output
#
# We need 2 pairs of W and b

# In[386]:


W2 = tf.Variable(initial_value=np.random.rand(state_size,num_classes),dtype=tf.float32)
b2 = tf.Variable(initial_value=np.random.rand(1,num_classes),dtype=tf.float32)


# Unpack

# In[387]:


labels_series = tf.unstack(batchY_placeholder, axis=1)


# Forward pass - Unroll the cell

# In[388]:


cell = tf.contrib.rnn.BasicLSTMCell(num_units=state_size)

states_series, current_state = tf.nn.dynamic_rnn(cell=cell,inputs=batchX_placeholder,dtype=tf.float32)


# In[389]:


states_series = tf.transpose(states_series,[1,0,2])


# Backward pass - Output

# In[390]:


last_state = tf.gather(params=states_series,indices=states_series.get_shape()[0]-1)
last_label = tf.gather(params=labels_series,indices=len(labels_series)-1)


# Backward pass - Output

# In[391]:


weight = tf.Variable(tf.truncated_normal([state_size,num_classes]))
bias = tf.Variable(tf.constant(0.1,shape=[num_classes]))


# Backward pass - Output

# In[392]:


prediction = tf.matmul(last_state,weight) + bias
prediction


# In[393]:


loss = tf.reduce_mean(tf.squared_difference(last_label,prediction))

train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)


# In[394]:


loss_list = []
test_pred_list = []

with tf.Session() as sess:

    tf.global_variables_initializer().run()

    for epoch_idx in range(num_epochs):

        print('Epoch %d' %epoch_idx)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * truncated_backprop_length
            end_idx = start_idx + truncated_backprop_length * batch_size


            batchX = xTrain[start_idx:end_idx,:].reshape(batch_size,truncated_backprop_length,num_features)
            batchY = yTrain[start_idx:end_idx].reshape(batch_size,truncated_backprop_length,1)

            #print('IDXs',start_idx,end_idx)
            #print('X',batchX.shape,batchX)
            #print('Y',batchX.shape,batchY)

            feed = {batchX_placeholder : batchX, batchY_placeholder : batchY}

            #TRAIN!
            _loss,_train_step,_pred,_last_label,_prediction = sess.run(
                fetches=[loss,train_step,prediction,last_label,prediction],
                feed_dict = feed
            )

            loss_list.append(_loss)



            ''' if(batch_idx % 50 == 0):        #print values
                print('Step %d - Loss: %.6f' %(batch_idx,_loss))'''

    #TEST


    for test_idx in range(len(xTest) - truncated_backprop_length):

        testBatchX = xTest[test_idx:test_idx+truncated_backprop_length,:].reshape((1,truncated_backprop_length,num_features))
        testBatchY = yTest[test_idx:test_idx+truncated_backprop_length].reshape((1,truncated_backprop_length,1))


        #_current_state = np.zeros((batch_size,state_size))
        feed = {batchX_placeholder : testBatchX,
            batchY_placeholder : testBatchY}

        #Test_pred contains 'window_size' predictions, we want the last one
        _last_state,_last_label,test_pred = sess.run([last_state,last_label,prediction],feed_dict=feed)
        test_pred_list.append(test_pred[-1][0]) #The last one


# In[395]:


print("ytest", len(yTest))
print("test_pred_list", len(test_pred_list))
plt.figure(figsize=(21,7))
plt.plot(yTest,label='Price',color='blue')
plt.plot(test_pred_list,label='Predicted',color='red')
plt.title('Price vs Predicted')
plt.legend(loc='upper left')
plt.show()


# In[397]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.title('Loss')
plt.scatter(x=np.arange(0,len(loss_list)), y=loss_list)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()


# In[ ]:
