import pandas
import codecs
import os
import uniout
import sys

train = pandas.read_csv(os.path.dirname(os.getcwd())+'/2_word2vec/21train_utf8.csv', header=None, encoding='utf-8' )
X_train, Y_train = list(train[1]), list(train[0])
test = pandas.read_csv(os.path.dirname(os.getcwd())+'/2_word2vec/21test_utf8.csv', header=None, encoding='utf-8' )
X_test, Y_test = list(test[1]), list(test[0])

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer

#nbc means naive bayes classifier
nbc_1 = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', MultinomialNB()),
])
nbc_2 = Pipeline([
    ('vect', HashingVectorizer(non_negative=True)),
    ('clf', MultinomialNB()),
])
nbc_3 = Pipeline([
    ('vect', TfidfVectorizer()),
    ('clf', MultinomialNB()),
])
nbcs = [nbc_1, nbc_2, nbc_3]

from sklearn.cross_validation import cross_val_score, KFold
from scipy.stats import sem
import numpy as np

def evaluate_cross_validation(clf, X, y, K):
    # create a k-fold croos validation iterator of k=5 folds
    cv = KFold(len(y), K, shuffle=True, random_state=0)
    # by default the score used is the one returned by score method of the estimator (accuracy)
    scores = cross_val_score(clf, X, y, cv=cv)
    print scores
    print ("Mean score: {0:.3f} (+/-{1:.3f})").format(
        np.mean(scores), sem(scores))
    
for nbc in nbcs:
    evaluate_cross_validation(nbc, X_train, Y_train, 5)
    
nbc_4 = Pipeline([
    ('vect', TfidfVectorizer(
                token_pattern=ur"\b[a-z0-9_\-\.]+[a-z][a-z0-9_\-\.]+\b",
    )),
    ('clf', MultinomialNB()),
])

#evaluate_cross_validation(nbc_4, X_train, Y_train, 5)

def get_stop_words():
    result = set()
    for line in open('stopwords_en.txt', ''r'').readlines():
        result.add(line.strip())
    return result

nbc_6 = Pipeline([
    ('vect', TfidfVectorizer(
#                stop_words=stop_words,
#                token_pattern=ur"\b[a-z0-9_\-\.]+[a-z][a-z0-9_\-\.]+\b",         
    )),
    ('clf', MultinomialNB(alpha=0.01)),
])
evaluate_cross_validation(nbc_6, X_train, Y_train, 5)

from sklearn import metrics
nbc_6.fit(X_train, Y_train)
print "Accuracy on training set:"
print nbc_6.score(X_train, Y_train)
print "Accuracy on testing set:"
print nbc_6.score(X_test,Y_test)
y_predict = nbc_6.predict(X_test)
print "Classification Report:"
print metrics.classification_report(Y_test,y_predict)
print "Confusion Matrix:"
print metrics.confusion_matrix(Y_test,y_predict)