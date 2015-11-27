import collections
import numpy as np
import csv
from utils import unicode_reader

class CorrectionHelper:
	def __init__(self, corrections, spoken, target):
		self.corrections = corrections
		self.spoken = spoken
		self.target = target

	def extractAndCleanCorrections(self):
		for index, value in enumerate(self.corrections):
			for char in value:
				if char is '<':
					first = char
				elif char is '>':
					pass
		return self.corrections

if __name__ == '__main__':
	data_src = 'data/lang-8-data.csv'
	reader = unicode_reader(open(data_src))
	for speaking, studying, incorrect, correct in reader:
		print(speaking, studying, incorrect, correct)