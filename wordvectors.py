from __future__ import unicode_literals, print_function
import numpy as np
import pickle
import gensim
import nltk
from utils import unpickleFile
from textblob import TextBlob
from nltkparsing import *
from nltk.tokenize import MWETokenizer
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer

PICKLE_PATH = 'data/all-data.pickle'

VEC_DIM = 150
NUM_BATCHES = 50

data = unpickleFile(PICKLE_PATH)

def testTranslation(entry):
    sentences = parseSentenceFeatures(entry)
    for sent in sentences:
        blob = TextBlob(sent)
        translate = blob.translate(to='en')

def gatherWordList(data=data):
    entry = data[900]['Entry']
    #print(entry)
    testTranslation(entry)
    words = parseWordsFromEntry(entry)
    sentences = parseSentenceFeatures(entry)
    words_sents = parseWordsFromSentences(sentences)
    model = gensim.models.Word2Vec(words)
    phrases = gensim.models.Phrases(words)

#gatherWordList(data)
    
def tokenFeatures(token, part_of_speech):
    if token.isdigit():
        yield "numeric"
    else:
        yield "token={}".format(token.lower())
        yield "token,pos={},{}".format(token, part_of_speech)
    if token[0].isupper():
        yield "uppercase_initial"
    if token.isupper():
        yield "all_uppercase"
    yield "pos={}".format(part_of_speech)

def bagOfWords(dataset):
    vec = CountVectorizer(max_features=1000)
    counts = vec.fit_transform(dataset)
    tfidf = TfidfTransformer(use_idf=False).fit(counts)
    x = tfidf.transform(counts)
    return x

def taggedTokenMatrix(tagged_tokens):
    raw = (tokenFeatures(token[0], token[1]) for token in tagged_tokens)
    hasher = FeatureHasher(input_type='string')
    sparse_matrix = hasher.transform(raw)
    return sparse_matrix

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