from __future__ import print_function
from time import time
from utils import unpickleFile, languages, evaluateClassifier
from wordvectors import bagOfWords
from SetProcessing import SetProcessing

from sklearn.linear_model import SGDClassifier, SGDRegressor, BayesianRidge
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import metrics

import numpy as np
import pickle

sp = SetProcessing()
targets = languages()

'''TODO'''
n_samples = 2000
n_features = 1000
n_languages = len(targets)

sgd_pipeline = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                            alpha=1e-3, n_iter=5, random_state=42)),])

print("Training on train set...")
train_data = sp.convertDataToList(sp.train)
train_entries, train_langs = sp.returnEntriesWithSpoken(train_data)
train_target = bagOfWords(train_entries)
sgd_pipeline = sgd_pipeline.fit(train_entries, train_langs)
testing = train_entries
predicted = sgd_pipeline.predict(testing) # 0.72870008767
#print(np.mean(predicted == train_langs))
print(metrics.classification_report(train_langs, predicted, target_names=train_langs))

#sgd_pipeline.fit(train_data, train_entries)
#predicted = text_clf.predict(docs_test)
#classifier = SVC.fit(tfidf, train_langs)
#classifier = LinearSVC.fit(x, labels)
#classifier = BayesianRidge.fit(x,)
#train_acc = classifier.score(x.toarray(), labels)
#print("Train accuracy: " + str(train_acc))

#print("Training on development set...")

#print("Training on test set....")

