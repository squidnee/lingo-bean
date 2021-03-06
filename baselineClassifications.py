from __future__ import print_function
from time import time
from utils import unpickleFile, languages, evaluateClassifier
from SetProcessing import SetProcessing

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import metrics
from sklearn.grid_search import GridSearchCV

import numpy as np
import pickle
import itertools
import random

sp = SetProcessing()

'''TODO'''
n_samples = 2000
n_features = 500
n_languages = len(languages())

def runSGDPipeline(entries, langs):
	t0 = time()
	sgd_pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(1,1), max_features=n_features)),
                      ('tfidf', TfidfTransformer(use_idf=True)),
                      ('clf', SGDClassifier(loss='squared_hinge', penalty='l2',
                                            alpha=0.001, n_iter=5, random_state=42))])

	vect = CountVectorizer(ngram_range=(1,1), max_features=n_features)
	X_train_counts = vect.fit_transform(entries)
	tfidf = TfidfTransformer(use_idf=True).fit(X_train_counts)
	X_train_tfidf = tfidf.fit_transform(X_train_counts)

	clf = SGDClassifier(loss='squared_hinge', penalty='l2', alpha=0.001, n_iter=5, random_state=42)
	clf.fit(X_train_tfidf, langs)

	X_new_counts = vect.transform(entries)
	X_new_tfidf = tfidf.transform(X_new_counts)
	predicted = clf.predict(X_new_tfidf.toarray())

	print(np.mean(predicted == langs))
	print(metrics.classification_report(langs, predicted, target_names=langs))
	print(metrics.confusion_matrix(langs, predicted))
	print("Took %s seconds." % (time()-t0))
	print("n_samples: %d, n_features: %d" % X_train_tfidf.shape)
	return sgd_pipeline

def runSVCPipeline(entries, langs):
	t0 = time()
	svc_pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(1,1), max_features=n_features)),
                      ('tfidf', TfidfTransformer(use_idf=True)),
                      ('clf', LinearSVC(dual=False, loss='squared_hinge', max_iter=100, random_state=42))])

	vect = CountVectorizer(ngram_range=(1,1), max_features=n_features)
	X_train_counts = vect.fit_transform(entries)
	tfidf = TfidfTransformer(use_idf=True).fit(X_train_counts)
	X_train_tfidf = tfidf.transform(X_train_counts)

	clf = LinearSVC(dual=False, loss='squared_hinge', max_iter=100, random_state=42)
	clf.fit(X_train_tfidf, langs)

	X_new_counts = vect.transform(entries)
	X_new_tfidf = tfidf.transform(X_new_counts)
	#dec = clf.decision_function([[1]])
	predicted = clf.predict(X_new_tfidf.toarray())

	print(np.mean(predicted == langs))
	print(metrics.classification_report(langs, predicted, target_names=langs))
	print(metrics.confusion_matrix(langs, predicted))
	print("Took %s seconds." % (time()-t0))
	print("n_samples: %d, n_features: %d" % X_train_tfidf.shape)
	return svc_pipeline

def runTreePipeline(entries, langs):
	t0 = time()
	tree_pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(1,1), max_features=n_features)),
                      ('tfidf', TfidfTransformer(use_idf=True)),
                      ('clf', DecisionTreeClassifier(max_features=n_features))])

	vect = CountVectorizer(ngram_range=(1,1), max_features=n_features)
	X_train_counts = vect.fit_transform(entries)
	tfidf = TfidfTransformer(use_idf=True).fit(X_train_counts)
	X_train_tfidf = tfidf.transform(X_train_counts)

	clf = DecisionTreeClassifier(max_features=n_features)
	clf.fit(X_train_tfidf, langs)

	X_new_counts = vect.transform(entries)
	X_new_tfidf = tfidf.transform(X_new_counts)
	predicted = clf.predict(X_new_tfidf.toarray())

	print(np.mean(predicted == langs))
	print(metrics.classification_report(langs, predicted, target_names=langs))
	print(metrics.confusion_matrix(langs, predicted))
	print("Took %s seconds." % (time()-t0))
	print("n_samples: %d, n_features: %d" % X_train_tfidf.shape)
	return tree_pipeline

def runRFPipeline(entries, langs):
	t0 = time()
	rf_pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(1,1), max_features=n_features)),
                      ('tfidf', TfidfTransformer(use_idf=True)),
                      ('clf', RandomForestClassifier(n_estimators=10))])

	vect = CountVectorizer(ngram_range=(1,1), max_features=n_features)
	X_train_counts = vect.fit_transform(entries)
	tfidf = TfidfTransformer(use_idf=True).fit(X_train_counts)
	X_train_tfidf = tfidf.fit_transform(X_train_counts)

	clf = RandomForestClassifier(n_estimators=10)
	clf.fit(X_train_tfidf, langs)

	X_new_counts = vect.transform(entries)
	X_new_tfidf = tfidf.transform(X_new_counts)
	predicted = clf.predict(X_new_tfidf.toarray())

	print(np.mean(predicted == langs))
	print(metrics.classification_report(langs, predicted, target_names=langs))
	print(metrics.confusion_matrix(langs, predicted))
	print("Took %s seconds." % (time()-t0))
	print("n_samples: %d, n_features: %d" % X_train_tfidf.shape)
	return rf_pipeline

def tuneParameters(pipeline, entries, langs):
	parameters = {'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
               'tfidf__use_idf': (True, False),
               'clf__alpha': (1e-2, 1e-3, 1.0),
               'clf__loss': ('hinge', 'squared_hinge')}
	gs_clf = GridSearchCV(pipeline, parameters, n_jobs=-1)
	gs_clf.fit(entries[:400], langs[:400])
	best_parameters, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])
	for param_name in sorted(parameters.keys()):
	    print("%s: %r" % (param_name, best_parameters[param_name]))

def trainOnTrainSet():
	print("Training on train set...")
	#train_data = sp.convertDataToList(sp.train)
	western, eastern = sp.organizeDataByRegion(sp.train)
	english, french, spanish = sp.organizeWesternLanguages(western)
	japanese, korean, mandarin = sp.organizeEasternLanguages(eastern)

	english = random.sample(english, 350)
	japanese = random.sample(japanese, 350)
	korean = random.sample(korean, 350)
	mandarin = random.sample(mandarin, 350)
	french = random.sample(french, 350)
	spanish = random.sample(spanish, 350)

	train_data = itertools.chain(english, french, spanish, japanese, korean, mandarin)
	train_data = list(train_data)
	random.shuffle(train_data)
	train_entries, train_langs = sp.returnEntriesWithSpoken(train_data)
	pipeline = runSGDPipeline(train_entries, train_langs)
	#pipeline = runSVCPipeline(train_entries, train_langs)
	#pipeline = runTreePipeline(train_entries, train_langs)
	#pipeline = runRFPipeline(train_entries, train_langs)
	#tuneParameters(pipeline, train_entries, train_langs)

def trainOnDevSet():
	print("Training on dev set...")
	#dev_data = sp.convertDataToList(sp.dev)
	western, eastern = sp.organizeDataByRegion(sp.dev)
	english, french, spanish = sp.organizeWesternLanguages(western)
	japanese, korean, mandarin = sp.organizeEasternLanguages(eastern)

	english = random.sample(english, 150)
	japanese = random.sample(japanese, 150)
	korean = random.sample(korean, 150)
	mandarin = random.sample(mandarin, 150)
	french = random.sample(french, 150)
	spanish = random.sample(spanish, 150)

	dev_data = itertools.chain(english, french, spanish, japanese, korean, mandarin)
	dev_data = list(dev_data)
	random.shuffle(dev_data)
	dev_entries, dev_langs = sp.returnEntriesWithSpoken(dev_data)
	#pipeline = runSGDPipeline(dev_entries, dev_langs)
	#pipeline = runSVCPipeline(dev_entries, dev_langs)
	pipeline = runTreePipeline(dev_entries, dev_langs)
	#pipeline = runRFPipeline(dev_entries, dev_langs)
	#tuneParameters(pipeline, dev_entries, dev_langs)

def trainOnTestSet():
	print("Training on test set...")
	#test_data = sp.convertDataToList(sp.test)
	western, eastern = sp.organizeDataByRegion(sp.test)
	english, french, spanish = sp.organizeWesternLanguages(western)
	japanese, korean, mandarin = sp.organizeEasternLanguages(eastern)

	english = random.sample(english, 50)
	japanese = random.sample(japanese, 50)
	korean = random.sample(korean, 50)
	mandarin = random.sample(mandarin, 50)
	french = random.sample(french, 50)
	spanish = random.sample(spanish, 50)

	test_data = itertools.chain(english, french, spanish, japanese, korean, mandarin)
	test_data = list(test_data)
	random.shuffle(test_data)
	test_entries, test_langs = sp.returnEntriesWithSpoken(test_data)
	pipeline = runSGDPipeline(test_entries, test_langs)
	#pipeline = runSVCPipeline(test_entries, test_langs)
	#pipeline = runTreePipeline(test_entries, test_langs)
	#pipeline = runRFPipeline(test_entries, test_langs)
	#tuneParameters(pipeline, test_entries, test_langs)

if __name__ == '__main__':
	trainOnTrainSet()
	#trainOnDevSet()
	#trainOnTestSet()