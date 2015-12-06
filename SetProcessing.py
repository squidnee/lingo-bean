from __future__ import print_function
from time import time
from utils import returnDatasets, generateExamples, asciistrip, makePrefixLangMapping
from collections import Counter
from textblob import TextBlob
'''
    This is a class that can be used for easy dataset examination, clustering,
    and parsing.
    '''

class SetProcessing():
	def __init__(self):
		self.train, self.dev, self.test = returnDatasets()

	def convertDataToList(self, dataset):
		t0 = time()
		alldata = []
		alldata_counter = 0
		for key, val in dataset.items():
			data = []
			alldata_counter += 1
			data.append(val['Speaking'])
			data.append(val['Studying'])
			data.append(val['Entry'])
			data.append(val['Incorrect'])
			data.append(val['Corrections'])
			alldata.append(data)
		print("There were %s entries sorted" % alldata_counter)
		print("Took %s seconds" % (time() - t0))
		return alldata

	def organizeDataByRegion(self, dataset):
		# Partitions the language by region (Western or Eastern).
		# Eastern languages: Japanese, Korean, Mandarin
		# Western languages: English, French, Spanish
		t0 = time()
		western_all = []; western_opts = ['English', 'French', 'Spanish']
		western_counter = 0
		eastern_all = []; eastern_opts = ['Japanese', 'Korean', 'Mandarin']
		eastern_counter = 0
		for key, val in dataset.items():
			if val['Speaking'] in western_opts:
				western = []
				western_counter += 1
				western.append(val['Speaking'])
				western.append(val['Studying'])
				western.append(val['Entry'])
				western.append(val['Incorrect'])
				western.append(val['Corrections'])
				western_all.append(western)
			elif val['Speaking'] in eastern_opts:
				eastern = []
				eastern_counter += 1
				eastern.append(val['Speaking'])
				eastern.append(val['Studying'])
				eastern.append(val['Entry'])
				eastern.append(val['Incorrect'])
				eastern.append(val['Corrections'])
				eastern_all.append(eastern)
		print("There were %s learners that natively speak a Western language" % western_counter)
		print("There were %s learners that natively speak an Eastern language" % eastern_counter)
		print("Took %s seconds" % (time() - t0))
		return western_all, eastern_all

	def organizeEasternLanguages(self, eastern):
		# Organizes a list of data from Eastern language speakers into each language.
		# Returns: Japanese, Korean, Mandarin lists (in that order)
		t0 = time()
		japanese = ['Japanese']; korean = ['Korean']; mandarin = ['Mandarin']
		jp_counter = 0; kr_counter = 0; zh_counter = 0
		for index, value in enumerate(eastern):
			if value[0] in japanese:
			 	jp_counter += 1
			 	japanese.append(eastern[index])
			elif value[0] in korean:
			 	kr_counter += 1
			 	korean.append(eastern[index])
			else:
			 	zh_counter += 1
			 	mandarin.append(eastern[index])
		print("There were %s learners that natively speak Japanese" % jp_counter)
		print("There were %s learners that natively speak Korean" % kr_counter)
		print("There were %s learners that natively speak Mandarin" % zh_counter)
		japanese.remove('Japanese')
		korean.remove('Korean')
		mandarin.remove('Mandarin')
		print("Took %s seconds" % (time() - t0))
		return japanese, korean, mandarin

	def organizeWesternLanguages(self, western):
		# Organizes a list of data from Western language speakers into each language.
		# Returns: English, French, Spanish lists (in that order)
		t0 = time()
		english = ['English']; french = ['French']; spanish = ['Spanish']
		en_counter = 0; fr_counter = 0; es_counter = 0
		for index, value in enumerate(western):
			if value[0] in english:
				en_counter += 1
				english.append(western[index])
			elif value[0] in french:
				fr_counter += 1
				french.append(western[index])
			else:
				es_counter += 1
				spanish.append(western[index])
		print("There were %s learners that natively speak English" % en_counter)
		print("There were %s learners that natively speak French" % fr_counter)
		print("There were %s learners that natively speak Spanish" % es_counter)
		english.remove('English')
		french.remove('French')
		spanish.remove('Spanish')
		print("Took %s seconds" % (time() - t0))
		return english, french, spanish

	def buildCorrectionPairs(self, datalist):
		# We organize the data by its incorrect pairs and its corrected counterparts.
		t0 = time()
		pairs = []
		for data in datalist:
			incorrect = []; corrections = []
			for i in data[3]:
				incorrect.append(i)
			for j in data[4]:
				corrections.append(j)
			print("There were %s incorrect with %s corrections" % (len(incorrect), len(corrections)))
			pair = (incorrect, corrections)
			pairs.append(pair)
		print("Took %s seconds" % (time() - t0))
		return pairs

	def buildSpeakingLearningPairs(self, datalist):
		t0 = time()
		western = ['English', 'French', 'Spanish']
		eastern = ['Japanese', 'Korean', 'Mandarin']
		western_counter = 0; eastern_counter = 0; other = 0
		pairs = []
		for data in datalist:
			studying = asciistrip(data[1])
			if studying in western: western_counter += 1
			if studying in eastern: eastern_counter += 1
			else: other += 1
			pair = (data[0], studying)
			pairs.append(pair)
		print("Took %s seconds" % (time() - t0))
		print("Out of %s native %s speakers, %s are learning a Western language, %s are"
		 " learning an Eastern language, and %s are learning an unincluded language." %
			 (len(pairs), pairs[0][0], western_counter, eastern_counter, other))
		return pairs

	def returnStudyingSet(self, datalist):
		t0 = time()
		learning = set(asciistrip(data[1]) for data in datalist)
		print("Took %s seconds" % (time() - t0))
		return learning

	def returnLanguageCounts(self, datalist):
		t0 = time()
		languages_learning = Counter()
		for data in datalist:
			studying = asciistrip(data[1])
			for s in studying.split():
				languages_learning[s] += 1
		print("Took %s seconds" % (time() - t0))
		return languages_learning

	def returnEntryVersusTarget(self, datalist):
		t0 = time()
		prefmap = makePrefixLangMapping()
		not_orig_lang = 0
		for data in datalist:
			blob = TextBlob(data[2])
			entrylang = blob.detect_language()
			islang = True
			for d in data[1].split():
				if entrylang not in prefmap: continue
				if prefmap[entrylang] == d: continue
				not_orig_lang += 1
		print("Took %s seconds" % (time() - t0))
		print("Of %s entries, there are %s entries written in a different language than specified" % 
			(len(datalist), not_orig_lang))

	def returnEntriesWithSpoken(self, datalist):
		t0 = time()
		entries = []; langs = []
		for data in datalist:
			entries.append(data[2])
			langs.append(data[0])
		print("Took %s seconds" % (time() - t0))
		return [entries, langs]

if __name__ == '__main__':
	sp = SetProcessing()
	western_speaking, eastern_speaking = sp.organizeDataByRegion(sp.test)
	japanese, korean, mandarin = sp.organizeEasternLanguages(eastern_speaking)
	english, french, spanish = sp.organizeWesternLanguages(western_speaking)

	entries = sp.convertDataToList(sp.train)