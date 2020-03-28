# -*- coding: utf-8 -*-
"""
Machine Translation with Python:
    
"""

# --------------------------------- Setup libraries
import helper
import numpy as np
import random
import re
import tensorflow as tf

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

from attention_decoder import AttentionDecoder


# ------------------------------------------ Read parallel text files
# read txt files
file = open("C:/Users/Chris/Documents/TensorFlow-Examples-Python/data/small_vocab_en.txt", "r")
en_sentences = file.readlines() 
file.close()

file = open("C:/Users/Chris/Documents/TensorFlow-Examples-Python/data/small_vocab_fr.txt", "r")
fr_sentences = file.readlines() 
file.close()


# ------------------------------------------ Tokenize and embed text



# ------------------------------------------ Define model
model = Sequential()
model.add(LSTM(150, input_shape=(n_timesteps_in, n_features), return_sequences=True))
model.add(AttentionDecoder(150, n_features))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])