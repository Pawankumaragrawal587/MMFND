#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import urllib.parse
import pandas as pd
import numpy as np
import time
from time import sleep


# In[2]:


data = pd.read_csv('Dataset/Initial_Data/Target/TICNN.csv')


# In[3]:


data


# In[4]:


data = data.iloc[:,[1,4]].values


# In[5]:


data


# In[6]:


t0 = time.time()

for i in range(len(data)):
    
    url = data[i][1]
    name = data[i][0] + '.jpg'
        
    try:
                
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36', "Upgrade-Insecure-Requests": "1","DNT": "1","Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8","Accept-Language": "en-US,en;q=0.5","Accept-Encoding": "gzip, deflate"}

        res = requests.get(url,headers=headers,timeout=7)

        with open('Dataset/Initial_Data/TargetImages/' + name, "wb") as f:
            f.write(res.content)

        print('Downloaded Image:',i)
    
    except:
        
        print('Could not Download Image:',i)

        sleep(1)
    
      
    print(i, 'Time elapsed:',time.time()-t0,'sec')
    
print('Average time per query:',(time.time()-t0)/len(data),'seconds.')

