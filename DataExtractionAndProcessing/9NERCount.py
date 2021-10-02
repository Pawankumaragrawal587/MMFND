#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import spacy
import time


# In[2]:


nlp = spacy.load('en_core_web_lg')

nlp.max_length = 4000000

# In[3]:


def get_ner(doc):
    
    doc = str(doc)

    if len(doc)>3000000:
        doc = doc[0:3000000]

    doc = nlp(doc)
    
    res = set()
    
    for ent in doc.ents:
        res.add(ent.text)
    
    return res


# In[4]:


target = pd.read_csv('Dataset/Initial_Data/Target/TICNN.csv')

target


# In[5]:


source = pd.read_csv('Dataset/Initial_Data/Source/TICNN.csv')

source


# In[6]:


source = source.drop(['Unnamed: 0', 'numSources', 
             'Source_url1', 'Image_url1', 'Source_reliability1', 
             'Source_url2', 'Image_url2', 'Source_reliability2',
             'Source_url3', 'Image_url3', 'Source_reliability3',
             'Source_url4', 'Image_url4', 'Source_reliability4',], axis=1)

source


# In[7]:


for i in range(10):
    
    target_entities = get_ner(target['Target_text'][i])
    
    print('################', i, '####################')
    print(target_entities)
    
    for j in range(1,5):
        
        source_entities = get_ner(source['Source_text'+str(j)][i])
        print(i,j,target_entities.intersection(source_entities))


# In[8]:


t0 = time.time()

for i in range(source.shape[0]):
    
    target_entities = get_ner(target['Target_text'][i])
        
    for j in range(1,5):
        
        source_entities = get_ner(source['Source_text'+str(j)][i])
        
        source['Source_text'+str(j)][i] = {'Common_entities': len(target_entities.intersection(source_entities)), 
                                           'Target_entities': len(target_entities)}
    
    print(i,'Time elapsed: ',time.time()-t0, 'Seconds')


# In[9]:


source


# In[10]:


source.to_csv('Dataset/Initial_Data/NER_TICNN.csv')

