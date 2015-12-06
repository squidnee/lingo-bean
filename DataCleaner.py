from __future__ import print_function
from utils import asciistrip, filepaths, languages, returnDatasets
from collections import Counter
from CorrectionHelper import CorrectionHelper

class DataCleaner:
	def __init__(self, corrections=None, spoken=None, target=None):
		self.train_path, self.dev_path, self.test_path = filepaths()
		self.trainset, self.devset, self.testset = returnDatasets()
		self.languages_known = Counter()
		self.languages_learning = Counter()
		self.languages = languages()
		self.ch = CorrectionHelper(corrections, spoken, target)

	def astrip(self, string):
		return asciistrip(string)

	def cleanCorrections(self):
		return self.ch.extractAndCleanCorrections()

if __name__ == '__main__':
	import utils
	dc = DataCleaner()