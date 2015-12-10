from __future__ import unicode_literals, print_function

from nltk.tokenize import MWETokenizer
import nltk
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction import FeatureHasher, DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import metrics
from gensim.models.word2vec import Word2Vec

from utils import returnDatasets, languages, isWestern
from textblob import TextBlob
from nltkparsing import *
from itertools import islice
from SetProcessing import SetProcessing

import numpy as np
import pickle
import gensim
import csv
import pandas as pd
from time import time

''' PARAMETERS '''
VEC_DIM = 150
NUM_BATCHES = 500
CONTEXT = 5
ALPHA = 0.01
FREQ_SKIP = 500
WORKERS = 4
MODEL_PATH = 'wordvecs/dim%s_%swindow_all' % (VEC_DIM, CONTEXT)

def w2vParams():
    return VEC_DIM, NUM_BATCHES, CONTEXT

def chunks(data, batch):
    it = iter(data)
    for i in range(0, len(data), batch):
        yield {k:data[k] for k in islice(it, batch)}

def openWordVecs():
    open(MODEL_PATH)

def w2vWestern(entry):
    return entry.split()

def w2vEastern(entry):
    import tinysegmenter as ts
    segmenter = ts.TinySegmenter()
    sentences = segmenter.tokenize(entry)
    return sentences

def w2vRetrieve(datalist, language_tag=None):
    if language_tag is None: 
        modelpath = MODEL_PATH
    else:
        modelpath = 'wordvecs/dim%s_window%s_%s' % (VEC_DIM, CONTEXT, language_tag)
    sp = SetProcessing()
    t0 = time()
    sentences = []
    counter = 0
    for value in datalist:
        if counter % 100 is 0: 
            print("currently looked at %s sets" % counter)
        if isWestern(value[sp.STUDYING]):
            sentences.append(w2vWestern(value[sp.ENTRY]))
            counter += 1
        else:
            sentences.append(w2vEastern(value[sp.ENTRY]))
            counter += 1
    wordvec = Word2Vec(sentences, size=VEC_DIM, window=CONTEXT, alpha=ALPHA, workers=WORKERS, min_count=FREQ_SKIP)
    wordvec.init_sims(replace=True)
    wordvec.save(modelpath)
    print("Saved word2vec model in %s seconds!" % (time()-t0))
    return wordvec

def bagOfWords(dataset):
    sp = SetProcessing()
    datalist = sp.convertDataToList(dataset)
    japanese, korean, mandarin = sp.organizeEasternLanguages(datalist)
    datalist = datalist[870:970]
    pairs = sp.buildSpeakingLearningPairs(datalist)
    print(pairs)
    entries = []
    langs = []
    korean = korean[:10]
    japanese = japanese[:10]

    for s in korean:
        datalist.append(s)
    for fr in japanese:
        datalist.append(fr)

    for data in datalist:
        entries.append(data[sp.ENTRY])
        langs.append(data[sp.SPEAKING])

    print(langs)

    vect = CountVectorizer()
    X_train_counts = vect.fit_transform(entries)
    tfidf = TfidfTransformer(use_idf=True).fit(X_train_counts)
    X_train_tfidf = tfidf.transform(X_train_counts)
    X_train_tfidf = X_train_tfidf.toarray()

    tree = SGDClassifier()
    tree.fit(X_train_tfidf, langs)
    result = tree.predict(X_train_tfidf)
    print(np.mean(result == langs))
    print(metrics.classification_report(langs, result, target_names=langs))

def findMeanWordVector(words, model, num_feats):
	# Finds the average of all the word vectors.
    featureVec = np.zeros((num_feats,),dtype="float32")
    num_words = 0
    index2word_set = set(model.index2word)
    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    featureVec = np.divide(featureVec,num_words)
    return featureVec

if __name__ == '__main__':
    train, dev, test = returnDatasets()
    bagOfWords(dev[:2])