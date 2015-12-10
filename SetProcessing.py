from __future__ import print_function
from time import time
from utils import returnDatasets, asciistrip, makePrefixLangMapping
from collections import Counter
from textblob import TextBlob
import random

'''
    This is a class that can be used for easy dataset examination, clustering,
    and parsing.
    '''

class SetProcessing():
	def __init__(self):
		self.train, self.dev, self.test = returnDatasets()
		self.SPEAKING = 0
		self.STUDYING = 1
		self.ENTRY = 2
		self.INCORRECT = 3
		self.CORRECTIONS = 4

	def convertDataToList(self, dataset):
		'''Converts a dictionary of dictionaries into a list of lists.'''
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

	def convertDictsToListOfDicts(self, datadict):
		dictlist = []
		for key, value in datadict.items():
			dictlist.append(value)
		return dictlist

	def mergeLists(self, train, dev, test=None):
		every = train
		for d in dev:
			every.append(d)
		if test is not None:
			for t in test:
				every.append(t)
		return every

	def returnSplitDatasets(self, dataset, split_size, speaking=True):
		t0 = time()
		english = ['English']; french = ['French']; spanish = ['Spanish']
		japanese = ['Japanese']; korean = ['Korean']; mandarin = ['Mandarin']
		if speaking: query = 'Speaking'
		else: query = 'Studying'
		for key, val in dataset.items():
			found = val[query]
			if found in english: english.append(val)
			elif found in french: french.append(val)
			elif found in spanish: spanish.append(val)
			elif found in japanese: japanese.append(val)
			elif found in korean: korean.append(val)
			elif found in mandarin: mandarin.append(val)
		english.remove('English'); korean.remove('Korean')
		french.remove('French'); mandarin.remove('Mandarin')
		spanish.remove('Spanish'); japanese.remove('Japanese')
		english = random.sample(english, split_size)
		french = random.sample(french, split_size)
		spanish = random.sample(spanish, split_size)
		japanese = random.sample(japanese, split_size)
		korean = random.sample(korean, split_size)
		mandarin = random.sample(mandarin, split_size)
		print("Took %s seconds" % (time()-t0))
		return english, french, spanish, japanese, korean, mandarin

	def organizeDataByRegion(self, dataset, speaking=True):
		'''Partitions the language by region (Western or Eastern).
		   Eastern languages: Japanese, Korean, Mandarin
		   Western languages: English, French, Spanish'''
		t0 = time()
		western_all = []; western_opts = ['English', 'French', 'Spanish']
		western_counter = 0
		eastern_all = []; eastern_opts = ['Japanese', 'Korean', 'Mandarin']
		eastern_counter = 0
		if speaking: query = 'Speaking'
		else: query = 'Studying'
		for key, val in dataset.items():
			if val[query] in western_opts:
				western = []
				western_counter += 1
				western.append(val['Speaking'])
				western.append(val['Studying'])
				western.append(val['Entry'])
				western.append(val['Incorrect'])
				western.append(val['Corrections'])
				western_all.append(western)
			elif val[query] in eastern_opts:
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

	def organizeEasternLanguages(self, eastern, speaking=True):
		'''Organizes a list of data from Eastern language speakers into each language.
		   Returns: Japanese, Korean, Mandarin lists (in that order)'''
		t0 = time()
		japanese = ['Japanese']; korean = ['Korean']; mandarin = ['Mandarin']
		jp_counter = 0; kr_counter = 0; zh_counter = 0
		if speaking: query = self.SPEAKING
		else: query = self.STUDYING
		for index, value in enumerate(eastern):
			if value[query] in japanese:
			 	jp_counter += 1
			 	japanese.append(eastern[index])
			elif value[query] in korean:
			 	kr_counter += 1
			 	korean.append(eastern[index])
			elif value[query] in mandarin:
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

	def organizeWesternLanguages(self, western, speaking=True):
		'''Organizes a list of data from Western language speakers into each language.
		   Returns: English, French, Spanish lists (in that order)'''
		t0 = time()
		english = ['English']; french = ['French']; spanish = ['Spanish']
		en_counter = 0; fr_counter = 0; es_counter = 0
		if speaking: query = self.SPEAKING
		else: query = self.STUDYING
		for index, value in enumerate(western):
			if value[query] in english:
				en_counter += 1
				english.append(western[index])
			elif value[query] in french:
				fr_counter += 1
				french.append(western[index])
			elif value[query] in spanish:
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
		'''Organizes the data by its incorrect pairs and its corrected counterparts.'''
		t0 = time()
		pairs = []
		for data in datalist:
			incorrect = []; corrections = []
			for i in data[self.INCORRECT]:
				incorrect.append(i)
			for j in data[self.CORRECTIONS]:
				corrections.append(j)
			pair = (incorrect, corrections)
			pairs.append(pair)
		print("Took %s seconds" % (time() - t0))
		return pairs

	def buildSpeakingLearningPairs(self, datalist):
		'''Pairs the spoken language of each element in the data list with the
		   language being studied by that user.'''
		t0 = time()
		western = ['English', 'French', 'Spanish']
		eastern = ['Japanese', 'Korean', 'Mandarin']
		western_counter = 0; eastern_counter = 0; other = 0
		pairs = []
		for data in datalist:
			studying = asciistrip(data[self.STUDYING])
			if studying in western: western_counter += 1
			if studying in eastern: eastern_counter += 1
			else: other += 1
			pair = (data[self.SPEAKING], studying)
			pairs.append(pair)
		print("Took %s seconds" % (time() - t0))
		print("Out of %s native %s speakers, %s are learning a Western language, %s are"
		 " learning an Eastern language, and %s are learning an unincluded language." %
			 (len(pairs), pairs[0][self.SPEAKING], western_counter, eastern_counter, other))
		return pairs

	def returnEntries(self, datalist):
		t0 = time()
		entries = list(data[self.ENTRY] for data in datalist)
		print("Took %s seconds to return entries" % (time()-t0))
		return entries

	def returnStudyingSet(self, datalist):
		'''Returns a set of all unique languages being studied by the users in this dataset.'''
		t0 = time()
		learning = set(asciistrip(data[self.STUDYING]) for data in datalist)
		print("Took %s seconds" % (time() - t0))
		return learning

	def returnLanguageCounts(self, datalist):
		'''Returns a counter that tallies up the languages being learned
		   and the number of people learning those languages.'''
		t0 = time()
		languages_learning = Counter()
		for data in datalist:
			studying = asciistrip(data[self.STUDYING])
			for s in studying.split():
				languages_learning[s] += 1
		print("Took %s seconds" % (time() - t0))
		return languages_learning

	def returnEntryVersusTarget(self, datalist):
		'''Some users write in a language that is different from their target language
		   (i.e. if they are practicing a language that they didn't specify that they were
		   learning, or if they are writing an entry in their native language asking someone
		   to translate something for them). This function counts how many of these instances
		   exist in the specified dataset.'''
		t0 = time()
		prefmap = makePrefixLangMapping()
		not_orig_lang = 0
		for data in datalist:
			blob = TextBlob(data[self.ENTRY])
			entrylang = blob.detect_language()
			islang = True
			for d in data[self.STUDYING].split():
				if entrylang not in prefmap: continue
				if prefmap[entrylang] == d: continue
				not_orig_lang += 1
		print("Took %s seconds" % (time() - t0))
		print("Of %s entries, there are %s entries written in a different language than specified" % 
			(len(datalist), not_orig_lang))

	def returnEntriesWithSpoken(self, datalist):
		'''Returns pairs of entries coupled with the language currently spoken. This is
		   what will be used for training.'''
		t0 = time()
		entries = []; langs = []
		for data in datalist:
			entries.append(data[self.ENTRY])
			langs.append(data[self.SPEAKING])
		print("Took %s seconds" % (time() - t0))
		return [entries, langs]

	def returnEntriesByRegionStudying(self, datalist, western=True):
		'''Returns pairs of entries coupled with the language being studied.'''
		t0 = time()
		if western is True:
			english = ['English']; french = ['French']; spanish = ['Spanish']
			en_counter = 0; fr_counter = 0; es_counter = 0
			for index, value in enumerate(datalist):
				if value[self.STUDYING] in english:
					english.append( [value[self.ENTRY], value[self.STUDYING]] )
					en_counter += 1
				elif value[self.STUDYING] in french:
					french.append( [value[self.ENTRY], value[self.STUDYING]] )
					fr_counter += 1
				elif value[self.STUDYING] in spanish:
					spanish.append( [value[self.ENTRY], value[self.STUDYING]] )
					es_counter += 1
			english.remove('English')
			french.remove('French')
			spanish.remove('Spanish')
			print("Took %s seconds" % (time() - t0))
			print("%s people studying English" % en_counter)
			print("%s people studying French" % fr_counter)
			print("%s people studying Spanish" % es_counter)
			return english, french, spanish
		else:
			japanese = ['Japanese']; korean = ['Korean']; mandarin = ['Mandarin']
			jp_counter = 0; kr_counter = 0; zh_counter = 0
			for index, value in enumerate(datalist):
				if value[self.STUDYING] in japanese:
			 		japanese.append( [value[self.ENTRY], value[self.STUDYING]] )
			 		jp_counter += 1
				elif value[self.STUDYING] in korean:
			 		korean.append( [value[self.ENTRY], value[self.STUDYING]] )
			 		kr_counter += 1
				elif value[self.STUDYING] in mandarin:
			 		mandarin.append( [value[self.ENTRY], value[self.STUDYING]] )
			 		zh_counter += 1
			japanese.remove('Japanese')
			korean.remove('Korean')
			mandarin.remove('Mandarin')
			print("%s people studying Japanese" % jp_counter)
			print("%s people studying Korean" % kr_counter)
			print("%s people studying Mandarin" % zh_counter)
			print("Took %s seconds" % (time() - t0))
			return japanese, korean, mandarin

if __name__ == '__main__':
	sp = SetProcessing()
	western, eastern = sp.organizeDataByRegion(sp.test, False)
	sp.returnEntriesByRegionStudying(western, True)
	sp.returnEntriesByRegionStudying(eastern, False)
	#japanese, korean, mandarin = sp.organizeEasternLanguages(eastern, False)
	#english, french, spanish = sp.organizeWesternLanguages(western, False)
	#print(sp.returnEntryVersusTarget(english))