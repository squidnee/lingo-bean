import collections
import numpy as np
import xml.etree.ElementTree as ET
from utils import unicode_reader

class CorrectionHelper:
	def __init__(self, corrections, spoken, target):
		self.corrections = corrections
		self.spoken = spoken
		self.target = target

		self.cleaned = list()
		self.classifications = list()

	# TODO : clean the sentences using the element tree, return that list.
	# TODO(?) : store (and?) classify the errors hidden in brackets.
	def extractAndCleanCorrections(self):
		for index, value in enumerate(self.corrections):
			print(ET.fromstring(value.encode('utf-8')).itertext())
		return self.corrections