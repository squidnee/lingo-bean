from __future__ import print_function
from utils import filepaths, retrieveDatasetsWithStats
from collections import Counter
import pickle
import bleach

class DataCleaner:
	def __init__(self):
		self.train_path, self.dev_path, self.test_path = filepaths()
		datasets, stats = retrieveDatasetsWithStats()
		self.trainset, self.devset, self.testset = datasets
		self.languages_known = Counter()
		self.languages_learning = Counter()

	def findLanguages(self):
		for x in self.testset.values():
			self.languages_known[x['Speaking']] += 1
			studying = x['Studying'].encode('utf-8').decode('ascii', 'ignore').strip()
			self.languages_learning[studying] += 1

if __name__ == '__main__':
	dc = DataCleaner()
	dc.findLanguages()
	print("Languages known: " + str(dc.languages_known))
	print("Languages learning: " + str(dc.languages_learning))