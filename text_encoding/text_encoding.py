# -*- coding: utf-8 -*-
"""
NLP using Python: 
    
    Word tokenization:
        
    Word Embeddings:
        Learn within model
        Pre-learn with word2vec routine
        Load external embeddings
        
"""

# --------------------------------- Setup libraries
import pandas as pd
import numpy as np
import random
import re

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding


# ------------------------------------------ Define the text corpus
# read data from csv
movie_reviews = pd.read_csv("C:/Users/Chris/Documents/TensorFlow-Examples-Python/text_encoding/IMDB Dataset.csv")

# inspect data obj
movie_reviews.info()
movie_reviews.describe()
movie_reviews.head()

# create train/test split
train_set, test_set = train_test_split(movie_reviews, test_size = 0.5)

# encode y as integers
le = LabelEncoder().fit(["negative", "positive"])
y_train = le.transform(train_set['sentiment'].values)
y_test = le.transform(test_set['sentiment'].values)

# ------------------------------------------ Clean text
# function to clean text
TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', text)

def clean_text(text): # text = movie_reviews['review']
    # define function for indiv sentences
    def clean_sen(sen):
        # Removing html tags
        sentence = remove_tags(sen)
        # Remove punctuations and numbers
        sentence = re.sub('[^a-zA-Z]', ' ', sentence)
        # Single character removal
        sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
        # Removing multiple spaces
        sentence = re.sub(r'\s+', ' ', sentence)
        return sentence
    # apply fun to vector of sentences
    text_new = []
    sentences = list(text)
    for sen in sentences:
        text_new.append(clean_sen(sen))
    return text_new

train_set_clean = train_set.copy() 
train_set_clean['review'] = clean_text(train_set['review'])

test_set_clean = test_set.copy() 
test_set_clean['review'] = clean_text(test_set['review'])


# ------------------------------------------ Tokenize and pad cleaned text
tokenize_obj = Tokenizer(oov_token = True)
tokenize_obj.fit_on_texts(train_set_clean['review'])

# tokenize
train_tokens = tokenize_obj.texts_to_sequences(train_set_clean['review'])
test_tokens = tokenize_obj.texts_to_sequences(test_set_clean['review'])

# get max sentance length and size of vocab
max_length = max([len(sen.split()) for sen in train_set_clean['review']])
vocab_size = len(tokenize_obj.word_index) + 1

# create x by padding tokenized vectors
x_train = pad_sequences(train_tokens, maxlen = max_length, padding = "post")
x_test = pad_sequences(test_tokens, maxlen = max_length, padding = "post")


# ------------------------------------------ learnt embedding
# hyperparameters
embedding_dim = 50
gru_units = 32
gru_dropout = 0.2
gru_recurrent_dropout = 0.2

batch_size = 32
epochs = 5

# model
model = Sequential()
model.add(Embedding(input_dim = vocab_size, 
                    output_dim = embedding_dim, 
                    input_length = max_length))
model.add(GRU(units = gru_units,
              dropout = gru_dropout ,
              recurrent_dropout = gru_recurrent_dropout))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss = "binary_crossentropy",
              optimizer = "adam",
              metrics = ["accuracy"])

#y_train = train_set['sentiment'].values
model.fit(x_train, 
          y_train, 
          batch_size = batch_size, 
          epochs = epochs)
# ------------------------------------------