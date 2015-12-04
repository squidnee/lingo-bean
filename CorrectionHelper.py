import collections
import numpy as np
import bleach
from utils import unicodeReader
from copy import deepcopy
#import xml.etree.ElementTree as ET

class CorrectionHelper:
	def __init__(self, corrections, spoken, target):
		self.corrections = corrections
		self.spoken = spoken
		self.target = target
		self.testing_src = 'data/testing-correction-helper.txt'

		self.cleaned = list()
		self.classifications = list()

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

	def runOneTest(self, corrections=None):
		if corrections is None: corrections = self.corrections
		with open(self.testing_src, 'w') as f:
			for correction in corrections:
				f.write(str(correction))
		print("Done writing to file.")
		f.close()
