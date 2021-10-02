#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from ast import literal_eval
from PIL import Image
import os


# In[2]:


data = pd.read_csv('Dataset/Initial_Data/ImageSimilarities_TICNN.csv')

data


# In[3]:


data = data.iloc[:,1:].values

data


# In[4]:


source_data = pd.read_csv('Dataset/Initial_Data/Source/TICNN.csv')

source_data


# In[5]:


source_data = source_data.iloc[:,[1,5,9,13,17]].values

source_data


# In[6]:


for i in range(len(data)):
    
    for j in range(1,5):
        
        data[i][j] = literal_eval(data[i][j])
        source_data[i][j] = literal_eval(source_data[i][j])
        
        for k in range(len(source_data[i][j])):
            
            flag = 1
            
            for check in range(0,min(6,len(data[i][j]))):
                
                if data[i][j][check]['Index'] == k:
                    
                    flag = 0
                    break
                    
            if flag:
                
                name = data[i][0] + '_' + str(j) + '_' + str(k) + '.jpg'
                
                if os.path.exists('Dataset/Initial_Data/SourceImages/TICNN/' + name):
                    
                    os.remove('Dataset/Initial_Data/SourceImages/TICNN/' + name)
                    
            else:
                    
                print('Keeping image', i,j,k)

