# Remember to include the import line(s) if you have use valid module(s) other than the one listed here
import pandas as pd
import numpy as np
import re
import keras

from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from tensorflow.keras import regularizers
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
# from keras.initializers import HeNormal
from keras.initializers import he_normal as HeNormal


def preprocessing(sentence):
    # remove html tags
    tag = re.compile(r'<[^>]+>')
    sentence = tag.sub('', sentence)
    # remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    # remove single char
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    # remove multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence


def get_X_y(df):
    '''
    TODO 1:
        - X = a list of processed review strings
        - preprocess all the instances in the dataset 
        - and store the processed list of sentence into variable X
    
    TODO 2: 
        - y = 1d binary numpy array
        - map the sentiment label into a binary value of 0 and 1 
        - and store the binary numpy array into variable y
    '''
    
    index = df.keys()
    X = []
    y = np.zeros(df.index.stop, int)
    
    for i in range(df.index.stop):
        if df[index[1]][i] == "positive": y[i] = 1      # index[1] = "sentiment"
        
        sentence = df[index[0]][i]                      # index[0] = "review"
        X.append(preprocessing(sentence))
        
    return X,y


def readglovefile(filepath):
    import gzip
    with gzip.open(filepath,'r') as f:
        content=f.readlines()
    return [i.decode('utf8') for i in content]


def formatEmbDict(filepath):
    '''
    TODO: 
        - return a dict variable wordDict with [word as key] and [embedding as value]
            ( i.e. calling wordDict['the'] will return a numpy array of [-0.038194, -0.24487, 0.72812, ..., 0.27062])

    '''
    wordDict = {}
    contents = readglovefile(filepath)
    
    for content in contents:
        # content: str
        splitted_content = content.split()                                      # Split the str according to spaces
        wordDict[splitted_content[0]] = np.array(splitted_content[1:], float)   # key: the first str, value: turn the other str into float np arrray
    
    return wordDict

def myModel(vocab_size, embedding_matrix, maxlen):
    '''
        TODO: construct a MLP
            - with an accuracy higher than 70% in our private test set
            - uses at least 2 and most 4 dense layers (including the output layer)
            - only uses the listed keras modules (Dense, Flatten, Activation, Dropout)
                (FYI: Dropout is randomly setting input units to 0 during training 
                    to prevent overfitting, details can be found in https://keras.io/api/layers/regularization_layers/dropout/.)
        # Remark: your model will be trained with 6 epoches under the same training setting as here (e.g. the same training set, training epoches, optimizer etc.) for evaluation, 
            you may need to set some argument in the dense layer to prevent overfitting causing poorer result
    '''
    
    model = Sequential()
    
    embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False)
    # In embedding_layer, 
    # the word index array for each instance is transformed to the GloVe embedding 
    # according to the embeddings matrix
    
    model.add(embedding_layer)
    model.add(Flatten(input_shape=embedding_matrix.shape[1:]))
    
    ### Hidden Layers
    num_of_hidden_neuron = int(np.sqrt(maxlen*100))
    initializar = HeNormal()
    
    # Hidden dense layer 1
    model.add(Dense(units=num_of_hidden_neuron, activation='relu', 
                    kernel_initializer=initializar,
                    kernel_regularizer=regularizers.l2(0.002)
    ))
    
    # Hidden dropour layer
    dropout_rate = 0.2
    model.add(Dropout(dropout_rate))
    
    # Hidden dense layer 2
    model.add(Dense(units=int(num_of_hidden_neuron/2), activation='relu', 
                    kernel_initializer=initializar,
                    kernel_regularizer=regularizers.l2(0.002)
    ))
    
    # Hidden dense layer 3
    model.add(Dense(units=int(num_of_hidden_neuron/2), activation='relu', 
                    kernel_initializer=initializar,
                    kernel_regularizer=regularizers.l2(0.002)
    ))
    
    # Output Layer
    model.add(Dense(units=1, activation='sigmoid'))
    
    return model