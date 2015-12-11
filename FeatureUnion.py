from __future__ import print_function
import collections, bleach, json, itertools
import numpy as np

from SetProcessing import SetProcessing
from utils import unicodeReader, returnDatasets, makeLangPrefixMapping
from utils import unpickleFile as uf
from textblob import TextBlob, Word
from collections import Counter
from time import time
from nltkparsing import *
from wordvectors import *
from tinysegmenter import TinySegmenter
from konlpy.tag import Mecab, Kkma
from rakutenma import RakutenMA

from nltk.tag.stanford import StanfordPOSTagger
from nltk.tag.stanford_segmenter import StanfordSegmenter
from nltk.classify.decisiontree import DecisionTreeClassifier
from nltk.metrics.distance import edit_distance
from nltk.util import ngrams as ng

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import metrics
from sklearn.metrics import classification_report
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD
from sklearn.base import BaseEstimator, TransformerMixin

class SyntacticStructExtraction:
	def __init__(self):
		self.english = ['English']
		self.french = ['French']
		self.spanish = ['Spanish']
		self.japanese = ['Japanese']
		self.korean = ['Korean']
		self.mandarin = ['Mandarin']
		self.lang_index = ['English', 'French', 'Spanish', 'Japanese', 'Korean','Mandarin']

		self.english_freqs_path = 'data/english_freqs.pickle'
		self.french_freqs_path = 'data/french_freqs.pickle'
		self.spanish_freqs_path = 'data/spanish_freqs.pickle'
		self.mandarin_freqs_path = 'data/mandarin_freqs.pickle'
		self.japanese_freqs_path = 'data/japanese_freqs.pickle'
		self.korean_freqs_path = 'data/korean_freqs.pickle'

	def fit(self, x, y=None):
		return self

	def averageSentenceLengthByLanguage(self, entries):
		t0 = time()
		total_entries = len(entries)
		averaged = 0
		for entry in entries:
			averaged += self.averageSentenceLength(entry)
		print("Took %s seconds to return the avg. sent. length by language." % (time()-t0))
		return float(averaged) / total_entries

	def averageSentenceLength(self, entry):
		'''Finds the average sentence length in chars.'''
		sentences = parseSentenceFeatures(entry)
		total_sents = len(sentences)
		all_words_total = 0
		for sent in sentences:
			all_words_total += len(sent)
		return float(all_words_total) / total_sents

	def averageNumberOfTokens(self, entries, eastern=True):
		'''Finds the average number of words in a sentence.'''
		t0 = time()
		entries_count = len(entries)
		wordcount = 0
		for entry in entries:
			if eastern:
				wordcount += len(TinySegmenter().tokenize(entry))
			else:
				wordcount += len(entry.split())
		print("Took %s seconds to return the avg. # of tokens per entry." % (time()-t0))
		print(float(wordcount) / entries_count)
		return float(wordcount) / entries_count

	def spellingErrors(self, sentences):
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

	def tagWordsInSentences(self, studying, entry):
		'''Tags the part of speech for each word.'''
		jar_path = 'stanford-postagger-full/stanford-postagger.jar'
		if studying in self.english:
			words = parseWordsFromEntry(entry)
			tagged_words = tagWords(words)
			return tagged_words
		elif studying in self.japanese or self.korean or self.mandarin:
			#segmenter = TinySegmenter()
			#words = segmenter.tokenize(entry)
			rm = RakutenMA()
			tagged_words = rm.tokenize(entry)
			#mecab = Mecab()
			#tagged_words = mecab.pos(entry)
			return tagged_words
		else:
			if studying in self.spanish:
				model_path = 'stanford-postagger-full/models/spanish.tagger'
				words = parseWordsFromEntry(entry)
			elif studying in self.french:
				model_path = 'stanford-postagger-full/models/french.tagger'
				words = parseWordsFromEntry(entry)
			postagger = StanfordPOSTagger(model_path, jar_path, encoding='utf8')
			tagged_words = postagger.tag(words)
			return tagged_words

	def findFrequencyOfSequence(self, tagged_words, ngrams=2):
		pos = [tag for word, tag in tagged_words]
		ngramlist = ng(pos, ngrams)
		freqs = Counter(n for n in ngramlist)
		return freqs

	def makeLanguageCounter(self, studying, entries):
		t0 = time()
		freqs_counter = Counter()
		for entry in entries:
			tagged_words = self.tagWordsInSentences(studying, entry)
			freqs = self.findFrequencyOfSequence(tagged_words)
			freqs_counter = freqs_counter + freqs
		print("Took %s seconds to make lang. counter" % (time()-t0))
		#print(freqs_counter)
		return {studying: freqs_counter}

	def makeCounter(self, counters):
		counter_n = 0
		holder = dict()
		for counter_dict in counters:
			lang, counter = counter_dict.popitem()
			counter_n += len(counter.values())
			holder[lang] = counter.most_common(100)
		return holder, counter_n

	def matchLangBySyntaxProb(self, holder, counter_n, studying, entry):
		observations = self.findFrequencyOfSequence(self.tagWordsInSentences(studying, entry))
		length = len(observations)
		observations = observations.most_common(int(length / 4))
		problang = studying
		curr_suma = 0
		for x, y in holder.items():
			for elem, num in y:
				for tag in observations:
					(first, second), count = tag
					if first or second in y:
						suma = float(count) / num
						if suma > curr_suma: 
							problang = x
							curr_suma = suma
		return entry, curr_suma, problang

	def transform_sentence_length(self, pairs):
		features = np.recarray(shape=(len(pairs),),
		dtype=[('spoken', object),('sentence_count', object)])
		for i, pair in enumerate(pairs):
			spoken, count = pair
			features['spoken'][i] = spoken
			features['sentence_count'][i] = count
		return features

class GiveawayFeatureExtraction:
	def __init__(self):
		self.clusters = 5

	def scanForOtherLanguages(self, words, target):
		'''Checks to see if other, non-specified languages are in the entry.'''
		langmap = makeLangPrefixMapping()
		langprefs = set()
		for word in words:
			detection = detect(word)
			if detection is not langmap[target]:
				langprefs.add( (word, detection) )
		return langprefs

class SocialFeatureExtraction:
	def __init__(BaseEstimator, TransformerMixin):
		pass

	def topicClustering(self, datalist, language_tag):
		w2v = w2vRetrieve(datalist, language_tag)
		print("Beginning kmeans clustering...")
		t0 = time()
		word_vectors = w2v.syn0
		num_clusters = int(word_vectors.shape[0] / 5)
		kmeans_clustering = KMeans(n_clusters=num_clusters)
		idx = kmeans_clustering.fit_predict(word_vectors)
		print("Finished kmeans in %s seconds" % (time()-t0))
		word_centroid_map = dict(zip( w2v.index2word, idx ))
		for cluster in range(0,10):
			print("\nCluster %s" % cluster)
			words = []
			vals = list(word_centroid_map.values())
			keys = list(word_centroid_map.keys())
			for i in range(len(vals)):
				if vals[i]==cluster:
					words.append(keys[i])
			print(words)

class CorrectionExtraction:
	def __init__(self):
		pass

	def fit(self, x, y=None):
		return self

	def errorEditDistance(self, pairs):
		'''Calculates the edit distance between two strings.'''
		t0 = time()
		total = 0; distances = 0
		for pair in pairs:
			incorrect, correct = pair
			for i, j in zip(incorrect, correct):
				total += 1
				distances += edit_distance(i, j)
		print("Took %s seconds to calc. edit distance" % (time()-t0))
		return float(distances) / total

class TokenFeatures(BaseEstimator, TransformerMixin):
	def fit(self, x, y=None):
		return self

	def transform_token_count(self, pairs):
		features = np.recarray(shape=len(pairs),),
		dtype=[('spoken', object),('token_count', object)]
		for i, pair in enumerate(pairs):
			spoken, count = pair
			features['spoken'][i] = spoken
			features['token_count'][i] = count
		return features

class FeatureGetter(BaseEstimator, TransformerMixin):
	def __init__(self):
		self.item = item

	def fit(self, x, y=None):
		return self

	def transform(self, item):
		return item

class POSFeatures(BaseEstimator, TransformerMixin):
	def fit(self, x, y=None):
		return self

	def transform(self, pairs):
		features = np.recarray(shape=(len(pairs),),
								dtype=[('entry', object),('pos_probability', object),('problang', object)])
		for i, pair in enumerate(pairs):
			entry, prob, problang = pair
			features['entry'][i] = entry
			features['pos_probability'][i] = prob
			features['problang'][i] = problang
		return features

class CorrectionFeatures(BaseEstimator, TransformerMixin):
	def __init__(obj):
		self.obj = obj

	def fit(self, x, y=None):
		return self

	def transform(self, pairs):
		features = np.recarray(shape=(len(pairs),),dtype=[('spoken', object),('edit_dist', object)])
		for i, pair in enumerate(pairs):
			spoken, count = pair
			features['spoken'][i] = spoken
			features['edit_dist'][i] = count
		return features


if __name__ == '__main__':
	train, dev, test = returnDatasets()

	''' Set up the classes. '''
	#gfe = GiveawayFeatureExtraction()
	#sfe = SocialFeatureExtraction()
	sse = SyntacticStructExtraction()
	ce = CorrectionExtraction()
	sp = SetProcessing()

	datalist = sp.convertDataToList(train)
	#dev = sp.convertDataToList(dev)
	#test = sp.convertDataToList(test)
	#merged = sp.mergeLists(train, dev, test)
	#english, french, spanish, japanese, korean, mandarin = sp.returnSplitDatasets(train, 5, False)
	'''Return the individual sets by native language.'''
	'''Takes approx. 1 second.'''
	print("Collecting test sets...")
	western_native, eastern_native = sp.organizeDataByRegion(train)
	english_native, french_native, spanish_native = sp.organizeWesternLanguages(western_native)
	japanese_native, korean_native, mandarin_native = sp.organizeEasternLanguages(eastern_native)

	'''Return the individual sets by language being studied.'''
	'''Takes approx. 1 second.'''
	western_learning, eastern_learning = sp.organizeDataByRegion(train, False)
	english_learning, french_learning, spanish_learning = sp.organizeWesternLanguages(western_learning, False)
	japanese_learning, korean_learning, mandarin_learning = sp.organizeEasternLanguages(eastern_learning, False)

	'''First feature: retrieve word frequencies. USE LEARNING SETS.'''
	'''Can take anywhere from 40 seconds to 7 minutes depending on the language and the set.'''
	#english_entries = sp.returnEntries(english_learning) # English works!
	#french_entries = sp.returnEntries(french_learning) # French works!
	#spanish_entries = sp.returnEntries(spanish_learning) # Spanish works!
	#mandarin_entries = sp.returnEntries(mandarin_learning) # Mandarin works!
	#korean_entries = sp.returnEntries(korean_learning) # Korean works!
	#japanese_entries = sp.returnEntries(japanese_learning)
	print("Gathering feature number one...")
	#freq_list = sse.makeLanguageCounter(japanese_learning[0][sp.STUDYING], japanese_entries)
	#pickle.dump(freq_list, open(sse.spanish_freqs_path, 'wb'))

	#pickle.dump(freq_list, open(sse.english_freqs_path, 'wb'))

	#pickle.dump(freq_list, open(sse.french_freqs_path, 'wb'))

	#pickle.dump(freq_list, open(sse.mandarin_freqs_path, 'wb'))

	#pickle.dump(freq_list, open(sse.japanese_freqs_path, 'wb'))

	#pickle.dump(freq_list, open(sse.korean_freqs_path, 'wb'))

	counter_list = [uf(sse.english_freqs_path), uf(sse.french_freqs_path), uf(sse.spanish_freqs_path),
	uf(sse.japanese_freqs_path), uf(sse.korean_freqs_path), uf(sse.mandarin_freqs_path)]

	holder, counter_n = sse.makeCounter(counter_list)

	lang_prob_path = 'data/langprobs_Train.pickle'
	#problang_pairs = []
	#finding_counter = 0
	#print("Finding language probability pairs...")
	#lang_prob_time = time()
	#for data in datalist:
	#	finding_counter += 1
	#	print("Have checked %s entries at %s" % (finding_counter, (time()-lang_prob_time)))
	#	entry, suma, problang = sse.matchLangBySyntaxProb(holder, counter_n, data[sp.STUDYING], data[sp.ENTRY])
	#	if problang is not None:
	#		problang_pairs.append( (entry, suma, problang) )
	#		pickle.dump(problang_pairs, open(lang_prob_path, 'wb'))
	
	'''Second feature: error distance between words. USE NATIVE SETS.'''
	'''Can take anywhere from 10 seconds to 9 minutes depending on the language and the set.'''
	print("Gathering feature number two...")
	mandarin_pairs = sp.buildCorrectionPairs(mandarin_native)
	korean_pairs = sp.buildCorrectionPairs(korean_native)
	japanese_pairs = sp.buildCorrectionPairs(japanese_native)
	english_pairs = sp.buildCorrectionPairs(english_native)
	french_pairs = sp.buildCorrectionPairs(french_native)
	spanish_pairs = sp.buildCorrectionPairs(spanish_native)

	english_dist= ce.errorEditDistance(english_pairs)
	mandarin_dist = ce.errorEditDistance(mandarin_pairs)
	korean_dist = ce.errorEditDistance(korean_pairs)
	japanese_dist = ce.errorEditDistance(japanese_pairs)
	french_dist = ce.errorEditDistance(french_pairs)
	spanish_dist = ce.errorEditDistance(spanish_pairs)

	dist_list = [('English', english_dist), ('French', french_dist), ('Spanish', spanish_dist),
	('Japanese', japanese_dist), ('Korean', korean_dist), ('Mandarin', mandarin_dist)]

	#ce.transform_edit_dist(dist_list)

	'''Third feature: average token length per sentence. USE NATIVE SETS.'''
	'''Takes about 5 seconds to 3.5 minutes depending on the language.'''
	print("Gathering feature number three...")
	english_entries = sp.returnEntries(english_native)
	french_entries = sp.returnEntries(french_native)
	spanish_entries = sp.returnEntries(spanish_native)
	mandarin_entries = sp.returnEntries(mandarin_native)
	korean_entries = sp.returnEntries(korean_native)
	japanese_entries = sp.returnEntries(japanese_native)

	mandarin_token = sse.averageNumberOfTokens(mandarin_entries)
	korean_token = sse.averageNumberOfTokens(korean_entries)
	japanese_token = sse.averageNumberOfTokens(japanese_entries)
	english_token = sse.averageNumberOfTokens(english_entries)
	french_token = sse.averageNumberOfTokens(french_entries)
	spanish_token = sse.averageNumberOfTokens(spanish_entries)

	tokens_list = [('English', english_token), ('French', french_token), ('Spanish', spanish_token),
	('Japanese', japanese_token), ('Korean', korean_token), ('Mandarin', mandarin_token)]

	#sse.transform_token_count(tokens_list)
	train_data = sp.convertDataToList(train)
	train_entries, train_langs = sp.returnEntriesWithSpoken(train_data)

	pipeline = Pipeline([
		('union', FeatureUnion(
			transformer_list=[

			# Pipeline for feature #1
			('feature_1', Pipeline([
				('selector', FeatureGetter(uf(lang_prob_path))),
				('word_freqs', POSFeatures()),
				('tfidf', TfidfVectorizer()),
				])),
			# Pipeline for feature 2
			('feature_2', Pipeline([
				('selector', FeatureGetter(dist_list)),
				('edit_dist', CorrectionFeatures()),
				('tfidf', TfidfVectorizer()),
				])),
			# Pipeline for feature 3
			('feature_3', Pipeline([
				('selector', FeatureGetter(tokens_list)),
				('token_count', TokenFeatures()),
				('tfidf', TfidfVectorizer()),
				])),
			# Pipeline for bag of words feature
			('bag_of_words', Pipeline([
				('selector', FeatureGetter(train_entries)),
				('vect', CountVectorizer(ngram_range=(1,1), max_features=500)),
				('tfidf', TfidfTransformer(use_idf=True)),
				])),
			# Weights for the features
			],
			transformer_weights = {
			'feature_1': 1.5,
			'feature_2': 0.5,
			'feature_3': 0.5,
			'feature_4': 1.5,
			},
		)),
		# Decision tree used as classifier
		('clf', DecisionTreeClassifier(max_features=500)),
	])

pipeline_pickle_path = 'data/pipeline.pickle'
pickle.dump(pipeline, open(pipeline_pickle_path, 'wb'))
print("About to run the pipeline...")
train_data = sp.convertDataToList(sp.train)
test_data = sp.convertDataToList(sp.train)
train_entries, train_langs = sp.returnEntriesWithSpoken(train_data)
test_entries, test_langs = sp.returnEntriesWithSpoken(test_data)
pipeline.fit(train_entries, train_langs)
y = pipeline.predict(train_entries)
print(classification_report(y, test_langs))