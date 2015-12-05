from __future__ import unicode_literals, print_function
from textblob import TextBlob
from sklearn.feature_extraction import DictVectorizer

def makeLangPrefixMapping():
	global lang_mapping
	lang_mapping = dict()
	return lang_mapping

def makePrefixLangMapping():
	global pref_mapping
	pref_mapping = dict()
	return pref_mapping

def featureExtractor(spoken, target, words, sentences, return_feats=False):
	global pref_mapping
	feats = {}
	prefixes = scanForMultipleLanguages(target, words)
	for pref in pref_mapping:
		feats["other_langs({0})".format(pref)] = 0
	for prefix in prefixes:
		feats["other_langs({0}".format(prefix)] = 1

	vec = DictVectorizer()
	pos_vectorized = vec.fit_transform(feats)
	pos_vectorized.toarray()
	
	if return_feats: return feats, vec
	else return vec

def scanForMultipleLanguages(target, words):
	global lang_mapping
	langprefs = set()
	for word in words:
		blob = TextBlob(words)
		if blob.detect_language() is not lang_mapping[target]:
			langprefs.add(blob.detect_language())
	return langprefs

if __name__ == '__main__':
	vec = DictVectorizer()