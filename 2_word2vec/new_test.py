#encoding=utf-8

import numpy as np
from sklearn import metrics
import pandas

import tensorflow as tf
import skflow
import uniout
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import gensim
### Training data
model = gensim.models.Word2Vec.load('cnki_wiki.model')
# Download dbpedia_csv.tar.gz from
# https://drive.google.com/folderview?id=0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M
# Unpack: tar -xvf dbpedia_csv.tar.gz

train = pandas.read_csv('7train.csv', header=None, encoding='utf-8' )
#X_train, y_train = train[2], train[0]
X_train, y_train = train[1], train[0]
test = pandas.read_csv('7test.csv', header=None, encoding='utf-8' )
X_test, y_test = test[1], test[0]
#X_test, y_test = test[2], test[0]

### Process vocabulary

MAX_DOCUMENT_LENGTH = 110

vocab_processor = skflow.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH)
X_train = np.array(list(vocab_processor.fit_transform(X_train)))
X_test = np.array(list(vocab_processor.transform(X_test)))

n_words = len(vocab_processor.vocabulary_)
print('Total words: %d' % n_words)

### Models

EMBEDDING_SIZE = 20
N_FILTERS = 10
WINDOW_SIZE = 20
FILTER_SHAPE1 = [WINDOW_SIZE, EMBEDDING_SIZE]
FILTER_SHAPE2 = [WINDOW_SIZE, N_FILTERS]
POOLING_WINDOW = 4
POOLING_STRIDE = 2

word_vec = np.zeros([21000,110,150])
print 1
for i in range(len(X_train)):
    for j in range(MAX_DOCUMENT_LENGTH):
        try:
            word_vec[i][j][:] = model[vocab_processor.vocabulary_.reverse(X_train[i][j])]
        except TypeError:
            continue
        except KeyError:
            continue
print 'Done'