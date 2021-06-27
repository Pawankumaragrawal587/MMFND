#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import requests
import urllib.parse
import pandas as pd
import numpy as np
import time
from time import sleep
from ast import literal_eval


# In[2]:


data = pd.read_csv('Dataset/Initial_Data/Source/TICNN.csv')


# In[3]:


data


# In[4]:


data = data.iloc[:,[1,2,5,9,13,17]].values


# In[5]:


data


# In[6]:


data.shape


# In[7]:


data[0][2][0] # It will return a sigle charecter (not complete image link) because list of links are stored as string


# In[8]:


# Converting list of imgae urls (stored as string) back to list

for i in range(len(data)):
    for j in range(2,6):
        data[i][j] = literal_eval(data[i][j])
    
data


# In[9]:


data[0][2][0]


# In[10]:


print('########## Process Starting... ##########')


t0 = time.time()

for i in range(5000,len(data)):
    for j in range(2,6):
        for k in range(len(data[i][j])):
            
            url = data[i][j][k]
            name = data[i][0] + '_' + str(j-1) + '_' + str(k) + '.jpg'

            if os.path.exists('Dataset/Initial_Data/SourceImages/TICNN/' + name):
                continue
            
            try:
                
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36', "Upgrade-Insecure-Requests": "1","DNT": "1","Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8","Accept-Language": "en-US,en;q=0.5","Accept-Encoding": "gzip, deflate"}
                
                res = requests.get(url,headers=headers,timeout=4)

                with open('Dataset/Initial_Data/SourceImages/TICNN/' + name, "wb") as f:
                    f.write(res.content)
                print('Downloaded Image',i,j,k)

            except:
                
                print('Unable to Download',i,j,k)

                sleep(1)

                    
    print(i, 'Time elapsed:',time.time()-t0,'sec')
    
print('Average time per query:',(time.time()-t0)/len(data),'seconds.')


print('########## Process Completed. ##########')