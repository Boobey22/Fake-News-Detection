#!/usr/bin/env python
# coding: utf-8

# # Pre-requisites:

# # 1. Tokenization

# In[1]:


import nltk
nltk.download()


# In[8]:


paragraph = """I have three visions for India. In 3000 years of our history, people from all over the world have come and invaded us, captured our lands, conquered our minds. From Alexander onwards, the Greeks, the Turks, the Moguls, the Portuguese, the British, the French, the Dutch, all of them came and looted us, took over what was ours. Yet we have not done this to any other nation."""
paragraph


# In[14]:


sentences = nltk.sent_tokenize(paragraph)
sentences


# In[15]:


words = nltk.word_tokenize(paragraph)
words


# # 2. Cleaning the text

# In[12]:


import re

#Remove punctuations from the String  
s = "!</> </>^may!!!% %the&& %$force@@ @be^^^&&!& </>*with@# you&&\ $%^"

s = re.sub(r'[^\w\s]','',s)
print(s)


# # 3. StopWords

# In[13]:


from nltk.corpus import stopwords

stop_words = stopwords.words('english')
print(stop_words)


# In[25]:


string = "Covid-19 pandemic has impacted many countries and what it did to economy is very stressful"


# In[26]:


tokens = nltk.word_tokenize(string)
tokens = [w for w in tokens if w not in stop_words]


# In[27]:


tokens


# # 4. Lemmatization

# In[30]:


from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
input_str = "been had done languages cities mice"

#Tokenize the sentence
input_str = nltk.word_tokenize(input_str)

#Lemmatize each word
for word in input_str:
    print(lemmatizer.lemmatize(word), end = ' ')


# # Clubbing them all:

# # Import Libraries and Dataset

# In[1]:


import pandas as pd
import numpy as np
import string


# In[20]:


df_text = pd.read_csv('news.csv', encoding = 'latin-1')
df_text


# In[21]:


df_text.columns = ['id', 'title', 'text', 'label']               
df_text.drop(['id', 'title'], axis = 1)          # Dropping out "id" and "title" from the data set as our target attribute is "label" and we will be working on "text"


# In[4]:


messages = df_text.copy()


# # Text Preprocessing, Lemmatization and Tokenization

# In[5]:


from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re


# In[6]:


lemmatizer = WordNetLemmatizer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', str(messages['text'][i]))    # Search for all non-letters
    review = review.lower()                                        # Replace them with spaces                                 
    review = review.split()
    
    review = [lemmatizer.lemmatize(word) for word in review if not word in stopwords.words('english')]   # Listing only the essential features(words)
    review = ' '.join(review)
    corpus.append(review)


# In[7]:


corpus[7]


# # TF/IDF Vectorizer

# In[8]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_v = TfidfVectorizer(max_features = 5000)
X = tfidf_v.fit_transform(corpus).toarray()


# In[9]:


X.shape


# In[10]:


y = messages['label']


# # Divide the set into Train and Test set

# In[11]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)


# In[12]:


tfidf_v.get_feature_names()[:20]


# In[13]:


count_df = pd.DataFrame(X_train, columns = tfidf_v.get_feature_names())    #To view the vectorized format of the feature extracted
count_df.head()


# # Multinomial Naive Baye's Theorem

# In[14]:


from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
from sklearn import metrics
import numpy as np
import itertools


# # Accuracy Score

# In[15]:


classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)
score = metrics.accuracy_score(y_test, pred)   # Calculating the accuracy
score

