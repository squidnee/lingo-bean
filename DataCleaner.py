from __future__ import print_function
from utils import asciistrip, filepaths, retrieveDatasetsWithStats
from collections import Counter
from CorrectionHelper import CorrectionHelper
import pickle
import bleach

class DataCleaner:
	def __init__(self, corrections=None, spoken=None, target=None):
		self.train_path, self.dev_path, self.test_path = filepaths()
		datasets, stats = retrieveDatasetsWithStats()
		self.trainset, self.devset, self.testset = datasets
		self.languages_known = Counter()
		self.languages_learning = Counter()
		self.ch = CorrectionHelper(corrections, spoken, target)

	def astrip(self, string):
		return asciistrip(string)

	def findLanguages(self):
		for x in self.testset.values():
			self.languages_known[x['Speaking']] += 1
			studying = asciistrip(x['Studying'])
			self.languages_learning[studying.split()[0]] += 1

	def cleanCorrections(self):
		return self.ch.extractAndCleanCorrections()

if __name__ == '__main__':
	dc = DataCleaner()
	dc.findLanguages()
	print("Languages known: " + str(dc.languages_known))
	print("Languages learning: " + str(dc.languages_learning))