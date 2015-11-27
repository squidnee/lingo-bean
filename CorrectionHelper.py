import collections
import numpy as np
import bleach
from utils import unicode_reader
from copy import deepcopy
#import xml.etree.ElementTree as ET

class CorrectionHelper:
	def __init__(self, corrections, spoken, target):
		self.corrections = corrections
		self.spoken = spoken
		self.target = target

		self.cleaned = list()
		self.classifications = list()

	# TODO: find out why <li> tags can't be removed
	def extractAndCleanCorrections(self):
		for index, value in enumerate(self.corrections):
			cleanedVal = bleach.clean(value, strip=True)
			if cleanedVal.find('[') is not -1: 
				cleanedVal = self.extractErrorClassification(cleanedVal)
			#print(cleanedVal)
			self.cleaned.append(cleanedVal)

	def extractErrorClassification(self, cleanedVal):
		bracket_open = cleanedVal.find('[')
		bracket_closed = cleanedVal.find(']')
		cleanedVal2 = deepcopy(cleanedVal)
		commentError = cleanedVal2[bracket_open+1:bracket_closed]
		commentErrorTuple = (commentError, 'Spoken: ' + str(self.spoken), 'Studying: ' + str(self.target))
		self.classifications.append(commentErrorTuple)
		new_str = str.replace(cleanedVal2, cleanedVal2[bracket_open-1:bracket_closed+1], ' ')
		return new_str
