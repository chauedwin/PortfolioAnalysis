#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


sp = pd.read_csv("constituents_csv.csv")['Symbol'].tolist()
nasdaq = pd.read_csv("nasdaq_screener.csv")['Symbol'].tolist()
ticks = sp + nasdaq


# In[24]:


import re
string = ''
with open('rawinfo.txt') as f:
    for i in range(1525):
        string = string + f.readline()
#print(string)
parsed = re.findall('[A-Z^\t*]+', string)
esgticks = [strings for strings in parsed if strings in ticks]
esgticks = list(set(esgticks))


# In[56]:


from pipeline import Portfolio

p = Portfolio(tickers = esgticks, start = '2016-07-01', end = '2021-07-01')
p.download()
p.clean_data(minmonths = 36)
p.compute_weights()
#p.compute_numstocks()
print(p.weights)

with open("esgoutput.txt", "w") as text_file:
    text_file.write(p.weights.to_string())

