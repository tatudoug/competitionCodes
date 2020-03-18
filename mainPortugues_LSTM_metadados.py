#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from zipfile import ZipFile
import re
from nltk.stem.snowball import SnowballStemmer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
from sklearn.linear_model import SGDClassifier
import logging
from textblob import TextBlob
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import gc
from sys import stdout
import argparse
from sklearn.metrics import recall_score
from sklearn.metrics import balanced_accuracy_score


# In[2]:


if (False):
    df_data = pd.read_csv('train.csv')
    df_data.head()


# In[41]:


df_test = pd.read_csv('test.csv')
df_test.head()
XT =df_test['title'].loc[df_test.language == 'portuguese']


# In[45]:


pd.concat([X,XT])


# # Saving a portuges file

# In[3]:


df_datap = pd.read_csv('train_portuguese.csv')


# In[4]:


len(df_datap)


# In[5]:


df_datap.head()


# In[6]:


len(df_datap.loc[df_datap.label_quality == 'reliable'])


# In[6]:


len(df_datap.category.unique())


# In[8]:


df_datap = df_datap.loc[df_datap.label_quality == 'reliable']


# In[7]:


# number of classes
len(df_datap['category'].unique())


# In[8]:


#Function to clean, tokenize, remove stop word, and not alphanumeric from data¶


# In[9]:


# Punctuation list
punctuations = re.escape('!"#%\'()*+,./:;<=>?@[\\]^_`{|}~')

# ##### #
# Regex #
# ##### #
re_remove_brackets = re.compile(r'\{.*\}')
re_remove_html = re.compile(r'<(\/|\\)?.+?>', re.UNICODE)
re_transform_numbers = re.compile(r'\d', re.UNICODE)
re_transform_emails = re.compile(r'[^\s]+@[^\s]+', re.UNICODE)
re_transform_url = re.compile(r'(http|https)://[^\s]+', re.UNICODE)
# Different quotes are used.
re_quotes_1 = re.compile(r"(?u)(^|\W)[‘’′`']", re.UNICODE)
re_quotes_2 = re.compile(r"(?u)[‘’`′'](\W|$)", re.UNICODE)
re_quotes_3 = re.compile(r'(?u)[‘’`′“”]', re.UNICODE)
re_dots = re.compile(r'(?<!\.)\.\.(?!\.)', re.UNICODE)
re_punctuation = re.compile(r'([,";:]){2},', re.UNICODE)
re_hiphen = re.compile(r' -(?=[^\W\d_])', re.UNICODE)
re_tree_dots = re.compile(u'…', re.UNICODE)
# Differents punctuation patterns are used.
re_punkts = re.compile(r'(\w+)([%s])([ %s])' %
                       (punctuations, punctuations), re.UNICODE)
re_punkts_b = re.compile(r'([ %s])([%s])(\w+)' %
                         (punctuations, punctuations), re.UNICODE)
re_punkts_c = re.compile(r'(\w+)([%s])$' % (punctuations), re.UNICODE)
re_changehyphen = re.compile(u'–')
re_doublequotes_1 = re.compile(r'(\"\")')
re_doublequotes_2 = re.compile(r'(\'\')')
re_trim = re.compile(r' +', re.UNICODE)


def clean_text(text):
    """Apply all regex above to a given string."""
    text = text.lower()
    text = text.replace('\xa0', ' ')
    text = re_tree_dots.sub('...', text)
    text = re.sub('\.\.\.', '', text)
    text = re_remove_brackets.sub('', text)
    text = re_changehyphen.sub('-', text)
    text = re_remove_html.sub(' ', text)
    text = re_transform_numbers.sub('0', text)
    text = re_transform_url.sub('URL', text)
    text = re_transform_emails.sub('EMAIL', text)
    text = re_quotes_1.sub(r'\1"', text)
    text = re_quotes_2.sub(r'"\1', text)
    text = re_quotes_3.sub('"', text)
    text = re.sub('"', '', text)
    text = re_dots.sub('.', text)
    text = re_punctuation.sub(r'\1', text)
    text = re_hiphen.sub(' - ', text)
    text = re_punkts.sub(r'\1 \2 \3', text)
    text = re_punkts_b.sub(r'\1 \2 \3', text)
    text = re_punkts_c.sub(r'\1 \2', text)
    text = re_doublequotes_1.sub('\"', text)
    text = re_doublequotes_2.sub('\'', text)
    text = re_trim.sub(' ', text)
    return text.strip()


# In[12]:


#txt.append(clean_text(line))
#sent_tokenizer.tokenize(line)


# In[16]:


for n in np.arange(5):
    f =  df_datap['title'].iloc[n] 
    print(f)
    print(clean_text(f))


# In[13]:


# testing the preprocessing


# In[14]:


'''for n in np.arange(30,35):
    f1 = clean_text(df_datap.title.iloc[n])
    #f2 = sent_tokenizer.tokenize(f1)
    print(n)
    print('original -', df_datap.title[n])
    print('cleaned -',f1)
    print('tokenized -',f2)
'''


# In[15]:


def len_text(text):
  if len(text.split())>0:
         return len(set(clean_text(text).split()))/ len(text.split())
  else:
         return 0


# In[16]:


#df_datap['len'] = df_datap['title'].apply(len_text)


# In[17]:


df_datap.head()


# In[18]:


# Metadados - subectivity and polarity (sentiment)
#df_news['subjectivity'] = df_news['text'].apply(subj_txt)
#df_news['polarity'] = df_news['text'].apply(polarity_txt)


# In[19]:


#len_text(df_datap['title'].iloc[1])


# In[20]:


#Deep Learning and Spacy Models


# In[17]:


file1 = open("glove_s50.txt","r") 
dicWord = {}
l = file1.readline()  #first line does not have information
for n in np.arange(929606):
    l = file1.readline()
    l = l.split()
    try:
        dicWord[l[0]] = np.array(l[1:],dtype=np.float16)
    except:
        print(n)
        print(l)
file1.close() 


# In[19]:


from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM, Embedding
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from keras.layers import Dense, Input, LSTM, Bidirectional, Activation, Conv1D, GRU, TimeDistributed
from keras.layers import Dropout, Embedding, GlobalMaxPooling1D, MaxPooling1D, Add, Flatten, SpatialDropout1D
from keras.layers import GlobalAveragePooling1D, BatchNormalization, concatenate
from keras.layers import Reshape, merge, Concatenate, Lambda, Average
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from sklearn.model_selection import train_test_split
import time
import pickle
from keras.models import Model
from keras.layers import Dense ,LSTM,concatenate,Input,Flatten,BatchNormalization, GRU


# # attention class

# In[23]:


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')
        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]
        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None
        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim
        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))
        if self.bias:
            eij += self.b
        eij = K.tanh(eij)
        a = K.exp(eij)
        if mask is not None:
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim


# In[51]:


X = df_datap['title']
#y =df_datap['category']
#encoder = LabelEncoder()
#y = encoder.fit_transform(y)
#Y = np_utils.to_categorical(y,dtype='bool')
##Create the tf-idf vector
vectorizer3 = TfidfVectorizer( min_df =2, max_df=0.2, max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = None, preprocessor=clean_text)


# In[46]:


vectorizer.fit(pd.concat([X,XT]))


# In[47]:


len(vectorizer.vocabulary_)


# In[49]:


vectorizer2.fit(X)


# In[50]:


len(vectorizer2.vocabulary_)


# In[ ]:


vectorizer3.fit(X)


# In[ ]:





# In[36]:


vectorizer.vocabulary_['máquina']


# In[43]:


del df_datap
gc.collect()


# In[26]:


Y.shape


# In[27]:


seed = 40
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)#,stratify =y)
vectorizer.fit(x_train)


# In[ ]:


with open('vectorizer_encoder.pickle', 'wb') as f:
    pickle.dump([vectorizer,encoder], f)

with open('train_testr.pickle', 'wb') as f:
    pickle.dump([x_train, x_test, y_train, y_test], f)


# In[28]:


if (False):
    with open('vectorizer_encoder.pickle', 'rb') as f:
        vectorizer,encoder = pickle.load(f)

    with open('train_test.pickle', 'rb') as f:
        x_train, x_test, y_train, y_test = pickle.load(f)


# In[29]:


word2idx = {word: idx for idx, word in enumerate(vectorizer.get_feature_names())}
tokenize = vectorizer.build_tokenizer()
preprocess = vectorizer.build_preprocessor()
 
def to_sequence(tokenizer, preprocessor, index, text):
    words = tokenizer(preprocessor(text))
    indexes = [index[word] for word in words if word in index]
    return indexes

X_train_sequences = [to_sequence(tokenize, preprocess, word2idx, x) for x in x_train]
print(X_train_sequences[0])


# In[30]:


MAX_SEQ_LENGHT=10

N_FEATURES = len(vectorizer.get_feature_names())
X_train_sequences = pad_sequences(X_train_sequences, maxlen=MAX_SEQ_LENGHT, value=N_FEATURES)
print(X_train_sequences[0])


# In[31]:


X_test_sequences = [to_sequence(tokenize, preprocess, word2idx, x) for x in x_test]
X_test_sequences = pad_sequences(X_test_sequences, maxlen=MAX_SEQ_LENGHT, value=N_FEATURES)


# In[32]:


EMBEDDINGS_LEN = 50

embeddings_index = np.zeros((len(vectorizer.get_feature_names()) + 1, EMBEDDINGS_LEN))
for word, idx in word2idx.items():
    try:
        embedding = dicWord[word]
        embeddings_index[idx] = embedding
    except:
        print(word)
        pass
      
print("EMBEDDINGS_LEN=", EMBEDDINGS_LEN)


# # Attention LSTM Model

# In[33]:


text_data = Input(shape=(MAX_SEQ_LENGHT,), name='text')
#meta_data = Input(shape=(1,), name = 'meta')
x = Embedding(len(vectorizer.get_feature_names()) + 1,
                    EMBEDDINGS_LEN,  # Embedding size
                    weights=[embeddings_index],
                    input_length=MAX_SEQ_LENGHT,
                    trainable=False)(text_data)
x1 = (LSTM(300, dropout=0.25, recurrent_dropout=0.25, return_sequences=True))(x)
x2 = Dropout(0.25)(x1)
x3 = Attention(MAX_SEQ_LENGHT)(x2)
x4 = Dense(256, activation='relu')(x3)
x5 = Dropout(0.25)(x4)
x6 = BatchNormalization()(x5)
##x7 = concatenate([x6, meta_data])
##x8 = Dense(150, activation='relu')(x7)
x8 = Dense(150, activation='relu')(x6)
x9 = Dropout(0.25)(x8)
x10 = BatchNormalization()(x9)
outp = Dense(len(set(y)), activation='softmax')(x10)

AttentionLSTM = Model(inputs=[text_data], outputs=outp)
AttentionLSTM.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

AttentionLSTM.summary()


# In[34]:


if(False):
    from keras.models import model_from_json

    # Model reconstruction from JSON file
    with open('model_architecture.json', 'r') as f:
        AttentionLSTM = model_from_json(f.read())

    # Load weights into the new model
    AttentionLSTM.load_weights('model_weights.h5')

    AttentionLSTM.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[80]:


AttentionLSTM.fit(X_train_sequences, y_train, 
          epochs=1, batch_size=128, verbose=1, 
          validation_split=0.1) # 20


# Save the weights
AttentionLSTM.save_weights('model_weights_att.h5')

# Save the model architecture
with open('model_architecture_att.json', 'w') as f:
    f.write(AttentionLSTM.to_json())
    
    

scores = AttentionLSTM.evaluate(X_test_sequences, y_test, verbose=1)
print("Accuracy:", scores[1])  #
print(("LSTM attention", scores[1]))
y_pred = AttentionLSTM.predict(X_test_sequences)
#np.argmax(y_pred,axis=1).shape
y_pred2 = np_utils.to_categorical(np.argmax(y_pred,axis=1),dtype='bool')
print("recall score LSTM attention", recall_score(y_test[:,:y_pred2.shape[1]], y_pred2, average='macro'))
print('balanced acc', balanced_accuracy_score(np.argmax(y_test,axis=1), np.argmax(y_pred,axis=1) ))


# In[79]:


print("recall score LSTM attention", recall_score(y_test[:,:y_pred2.shape[1]], y_pred2, average='macro'))
print('balanced acc', balanced_accuracy_score(np.argmax(y_test,axis=1), np.argmax(y_pred,axis=1) ))


# In[55]:


y_pred = AttentionLSTM.predict(X_test_sequences)


# In[41]:


np.argmax(y_pred,axis=1).shape


# In[44]:


y_pred2 = np_utils.to_categorical(np.argmax(y_pred,axis=1),dtype='bool')


# In[48]:


y_pred2.shape


# In[50]:


print("recall score LSTM attention", recall_score(y_test[:,:y_pred2.shape[1]], y_pred2, average='macro'))


# In[56]:


from sklearn.metrics import balanced_accuracy_score


# In[63]:


balanced_accuracy_score(np.argmax(y_test,axis=1), np.argmax(y_pred,axis=1))
#np.argmax(y_test,axis=1).shape


# In[54]:


min(y)


# In[61]:


y_pred


# In[ ]:




