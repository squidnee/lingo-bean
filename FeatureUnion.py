from __future__ import print_function
import collections
import numpy as np
import bleach
from SetProcessing import SetProcessing
from utils import unicodeReader, returnDatasets
from copy import deepcopy
from textblob import TextBlob, Word
from time import time
from nltkparsing import *

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import metrics
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD

class CorrectionExtraction:
	def __init__(self, corrections, spoken, target):
		self.corrections = corrections
		self.spoken = spoken
		self.target = target

		self.cleaned = list()
		self.classifications = list()
		self.clusters = 4

	def extractAndCleanCorrections(self):
		for index, value in enumerate(self.corrections):
			cleanedVal = bleach.clean(value, strip=True)
			if cleanedVal.find('[') is not -1: 
				cleanedVal = self.extractErrorClassification(cleanedVal)
			#print(cleanedVal)
			self.cleaned.append(cleanedVal)
		return self.cleaned

	def extractErrorClassification(self, cleanedVal):
		bracket_open = cleanedVal.find('[')
		bracket_closed = cleanedVal.find(']')
		cleanedVal2 = deepcopy(cleanedVal)
		commentError = cleanedVal2[bracket_open+1:bracket_closed]
		commentErrorTuple = (commentError, 'Spoken: ' + str(self.spoken), 'Studying: ' + str(self.target))
		self.classifications.append(commentErrorTuple)
		new_str = str.replace(cleanedVal2, cleanedVal2[bracket_open-1:bracket_closed+1], ' ')
		return new_str

	def collectSpellingErrors(self, sentences):
		spelling_errors_counter = 0
		misspelled_words = []
		for sent in sentences:
			blob = TextBlob(sent)
			correction = blob.correct()
			if correction is not sent:
				for word in sent.split():
					if word not in correction:
						misspelled_words.append(word)
						spelling_errors_counter += 1
		print("There were %s spelling errors." % spelling_errors_counter)
		return misspelled_words

	def confirmSpellcheck(tokens):
		spellcheck = []
		for token in tokens:
			w = Word(token)
			spellcheck.append(w.spellcheck())
		return spellcheck

	def clusterCorrections(self, data, clusters):
		t0 = time()
		vectorizer = TfidfVectorizer(max_features=100)
		X = vectorizer.fit_transform(data)
		svd = TruncatedSVD()
		normalizer = Normalizer(copy=False)
		lsa = make_pipeline(svd, normalizer)
		X = lsa.fit_transform(X)
		explained_variance = svd.explained_variance_ratio_.sum()
		print(explained_variance)
		batchkm = MiniBatchKMeans(n_clusters=clusters, init='k-means++', n_init=1,
                         init_size=1000, batch_size=1000)
		batchkm.fit(X)
		terms = vectorizer.get_feature_names()
    	for i in range(true_k):
        	print("Cluster %d:" % i, end='')
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind], end='')
        print("Finished in %s seconds" % (time() - t0))

class SyntacticStructExtraction:
	def __init__(self):
		pass

	def tagWordsInSentences(self, words):
		# Translate word. If english:
		tagged_words = tagWords(words)

	def guessLanguageByStruct(self):
		pass

class GiveawayFeatureExtraction:
	def __init__(self):
		pass

	def findUnevenCharacters(self):
		pass

	def scanForOtherLanguages(self):
		pass

if __name__ == '__main__':
	train, dev, test = returnDatasets()
	sp = SetProcessing()
	datalist = sp.convertDataToList(train)
	datalist = datalist[700:800]
	for t in datalist:
		ce = CorrectionExtraction(t[sp.CORRECTIONS], t[sp.SPEAKING], t[sp.STUDYING])
		ce.extractAndCleanCorrections()
		print(ce.classifications)