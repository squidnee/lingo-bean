import numpy as np
import pickle
from utils import unpickleFile
import gensim
import nltk
from textblob import TextBlob
from nltkparsing import *
from nltk.tokenize import MWETokenizer

entry_src = 'data/lang-8-entries.txt'
entry_bin = 'data/lang-8-entries.bin'
entry_clusters = 'data/lang-8-entries-clusters.txt'
PICKLE_PATH = 'data/all-data.pickle'

VEC_DIM = 150
NUM_BATCHES = 50

data = unpickleFile(PICKLE_PATH)

def gatherWordList(data=data):
    entry = data[982]['Entry']
    words = parseWordsFromEntry(entry)
    sentences = parseSentenceFeatures(entry)
    for sentence in sentences:
        b = TextBlob(sentence)
        print(b.correct())
    words_sents = parseWordsFromSentences(sentences)
    model = gensim.models.Word2Vec(words)
    phrases = gensim.models.Phrases(words)

gatherWordList(data)

def implementBagOfWords():
	pass

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