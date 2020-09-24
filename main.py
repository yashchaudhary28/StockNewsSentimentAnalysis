# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 23:33:03 2020

@author: yashc
"""

#Importing the required libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.request import urlopen, Request 
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


#Specifying url from where data is to be scrapped.
#tickers are names of various stocks on which we need to scrap data.
finviz_url = "https://finviz.com/quote.ashx?t="
tickers = ['AMZN', 'FB', 'GOOG', 'MSFT']

#In this step we are collecting data from the website using request and
# parsing finviz data using beautiful soup.
news_tables = {}
for ticker in tickers:
    url = finviz_url + ticker
    
    req = Request(url, headers = {'user-agent' : 'my-app'})
    response = urlopen(req)
    
    html = BeautifulSoup(response, 'html')
    news_table = html.find(id = 'news-table')
    news_tables[ticker] = news_table

#Now, we are taking all the parsed data that is saved in the dictionary and
#extract out only news headline data for all the stocks.
parsed_data = []

for ticker, news_table in news_tables.items():
    
    for row in news_table.findAll('tr'):
        
        title = row.a.text
        date_data = row.td.text.split()
        
        if len(date_data) == 1:
            time = date_data[0]
        else:
            time = date_data[1]
            date = date_data[0]
            
        parsed_data.append([ticker, date, time, title])

#Converting the list data to dataframe
df = pd.DataFrame(parsed_data, columns = ['Ticker', 'Date', 'Time', 'Title'])

#Applying sentiment analysis on news headlines and getting compond score.

vader = SentimentIntensityAnalyzer()
df['compound'] = df['Title'].apply(lambda x: vader.polarity_scores(x)['compound'])
df['Date'] = pd.to_datetime(df.Date).dt.date

plt.figure(figsize = (10, 8))
mean_df = df.groupby(['Ticker', 'Date']).mean()
mean_df = mean_df.unstack()
mean_df = mean_df.xs('compound', axis = 'columns').transpose()
mean_df.plot(kind = 'bar')
plt.show()










