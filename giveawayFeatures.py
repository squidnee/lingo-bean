from __future__ import unicode_literals, print_function
from textblob import TextBlob
from sklearn.feature_extraction import DictVectorizer
from utils import makePrefixLangMapping, makeLangPrefixMapping

def featureExtractor(spoken, target, words, sentences, return_feats=False):
	prefmap = makePrefixLangMapping()
	feats = {}
	prefixes = scanForMultipleLanguages(target, words)
	for pref in prefmap:
		feats["other_langs({0})".format(pref)] = 0
	for prefix in prefixes:
		feats["other_langs({0}".format(prefix)] = 1

	vec = DictVectorizer()
	pos_vectorized = vec.fit_transform(feats)
	pos_vectorized.toarray()
	
	if return_feats: return feats, vec
	else: return vec

def scanForMultipleLanguages(target, words):
	langmap = makeLangPrefixMapping()
	langprefs = set()
	for word in words:
		blob = TextBlob(words)
		detect = blob.detect_language()
		if detect is not langmap[target]:
			langprefs.add(detect)
	return langprefs

if __name__ == '__main__':
	vec = DictVectorizer()