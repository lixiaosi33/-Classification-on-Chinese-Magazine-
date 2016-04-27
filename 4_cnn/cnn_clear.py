#encoding=utf-8

import numpy as np
from sklearn import metrics
import pandas

import tensorflow as tf
import skflow
import uniout
import sys
reload(sys)
import os
sys.setdefaultencoding('utf-8')
### Training data
# Download dbpedia_csv.tar.gz from
# https://drive.google.com/folderview?id=0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M
# Unpack: tar -xvf dbpedia_csv.tar.gz

train = pandas.read_csv(os.path.dirname(os.getcwd())+'/2_word2vec/19train_utf.csv', header=None, encoding='utf-8' )
#X_train, y_train = train[2], train[0]
X_train, y_train = train[1], train[0]
test = pandas.read_csv(os.path.dirname(os.getcwd())+'/2_word2vec/19test_utf.csv', header=None, encoding='utf-8' )
X_test, y_test = test[1], test[0]
#X_test, y_test = test[2], test[0]

### Process vocabulary

MAX_DOCUMENT_LENGTH = 100

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

def cnn_model(X, y):
    """2 layer Convolutional network to predict from sequence of words
    to a class."""
    # Convert indexes of words into embeddings.
    # This creates embeddings matrix of [n_words, EMBEDDING_SIZE] and then
    # maps word indexes of the sequence into [batch_size, sequence_length,
    # EMBEDDING_SIZE].
    word_vectors = skflow.ops.categorical_variable(X, n_classes=n_words,
        embedding_size=EMBEDDING_SIZE, name='words')
    word_vectors = tf.expand_dims(word_vectors, 3)
    with tf.variable_scope('CNN_Layer1'):
        # Apply Convolution filtering on input sequence.
        conv1 = skflow.ops.conv2d(word_vectors, N_FILTERS, FILTER_SHAPE1, padding='VALID')
        # Add a RELU for non linearity.
        conv1 = tf.nn.relu(conv1)
        # Max pooling across output of Convlution+Relu.
        pool1 = tf.nn.max_pool(conv1, ksize=[1, POOLING_WINDOW, 1, 1], 
            strides=[1, POOLING_STRIDE, 1, 1], padding='SAME')
        # Transpose matrix so that n_filters from convolution becomes width.
        pool1 = tf.transpose(pool1, [0, 1, 3, 2])
    with tf.variable_scope('CNN_Layer2'):
        # Second level of convolution filtering.
        conv2 = skflow.ops.conv2d(pool1, N_FILTERS, FILTER_SHAPE2,
            padding='VALID')
        # Max across each filter to get useful features for classification.
        pool2 = tf.squeeze(tf.reduce_max(conv2, 1), squeeze_dims=[1])
    # Apply regular WX + B and classification.
    return skflow.models.logistic_regression(pool2, y)


classifier = skflow.TensorFlowEstimator(model_fn=cnn_model, n_classes=20,
    steps=100, optimizer='Adam', learning_rate=0.01, continue_training=True)

# Continuesly train for 1000 steps & predict on test set.
while True:
    classifier.fit(X_train, y_train, logdir='/tmp/tf_examples/word_cnn')
    score = metrics.accuracy_score(y_test, classifier.predict(X_test))
    print('Accuracy: {0:f}'.format(score))

