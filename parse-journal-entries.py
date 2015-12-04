#!/usr/bin/python
# coding: utf8
from __future__ import unicode_literals, print_function
import sys
import collections
import requests, csv
from bs4 import BeautifulSoup, NavigableString
from CorrectionHelper import CorrectionHelper
from pylab import *
import sqlite3 as sql
import pandas as pd
import pandas.io.sql as pdsql
import pickle

try:
    import timeoutsocket
    timeoutsocket.setDefaultSocketTimeout(10)
except ImportError:
    pass

PICKLE_FILE =  "data/all-data-3.pickle"
data_src = 'data/lang-8-url-201012.txt'
new_src = 'data/lang-8-url-cleaned-2.txt'

# Creates 'cleaned' text file with all duplicate URLs removed. Can comment out once file has been created.
# def deleteDuplicateURLs(data_src, new_src):
# 	bad_suffix = ['.1/']
# 	with open(data_src, 'r') as oldfile, open(new_src, 'w') as newfile:
# 		for url in oldfile:
# 			if not any(suffix in url for suffix in bad_suffix):
# 				newfile.write(url)

def mineLearnerData(url):
	resp = requests.get(url)
	raw_text = resp.text
	soup = BeautifulSoup(raw_text, 'html.parser', from_encoding="gb18030")
	for s in soup.findAll('script'):
		s.replaceWith('')
	try:
		entry = soup.select('div#body_show_ori')[0].text
	except IndexError:
		entry = 'Null'
	try:
		speaking = soup.select('li.speaking')[0].text
	except IndexError:
		speaking = 'Null'
	try:
		studying = soup.select('li.studying')[0].text
	except IndexError:
		studying = 'Null'
	try:
		incorrect, correct = returnCorrectedSets(soup)
	except IndexError:
		incorrect = 'Null'; correct = 'Null'
	return speaking, studying, entry, incorrect, correct

# Returns a set of incorrect sentences and their corrected forms, as deemed by native speakers.
# TODO: pair incorrect sentences with corrected counterparts
def returnCorrectedSets(soup):
		wrongSet = set(); correctSet = set()
		for incorrect in soup.select('div.correction_box ul.correction_field li.incorrect'):
			wrongSet.add(incorrect.text)
		for correct in soup.select('div.correction_box ul.correction_field li.corrected.correct'):
			correctSet.add(correct)
		return wrongSet, correctSet

# TODO
def createPickledDatasets(list_of_dicts, pickle_file=PICKLE_FILE):
	output = open(pickle_file, 'wb')
	pickle.dump(list_of_dicts, output)
	output.close()
	#readPickleTest(pickle_file)

def divideIntoDatasets(samples):
	# Creates 70% - 15% - 15% train/dev/test datasets.
	num = len(samples)
	train_src = "/data/TrainDataset_" + str(num)
	dev_src = "/data/DevDataset_" + str(num)
	test_src = "/data/TestDataset_" + str(num)
	#train_samples
	#dev_samples
	#test_samples
	#pickle.dump()

def readPickleTest(pickle_file=PICKLE_FILE):
	f = open(pickle_file, 'rb')
	my_list_of_dicts = pickle.load(f)
	f.close()

def debugPickle(filename):
    try:
        f = open(filename, 'r')
    except IOError:
        print('cannot open')
    else:
        print('has', len(f.readlines()), 'lines')
        f.close()

def runSimpleTests(new_src=new_src, pickle_file=PICKLE_FILE):
	f = open(new_src, 'r')
	total_in_set = 0
	num_unacceptable = 0
	total_learner_data = dict()
	for index, url in enumerate(f.readlines()):
		if index <= 15673: continue
		try:
			speaking, studying, entry, incorrect, correct = mineLearnerData(url)
			ch = CorrectionHelper(correct, speaking, studying)
			cleanedCorrect = ch.extractAndCleanCorrections()
			learner_data = {'Speaking': speaking, 'Studying': studying, 'Entry': entry, 'Incorrect': incorrect, 'Corrections': cleanedCorrect}
			total_learner_data[index] = learner_data
			createPickledDatasets(total_learner_data, pickle_file)
			total_in_set += 1
			print('Total found: ' + str(index), ' Number in set: ' + str(total_in_set))
		except IOError as e:
			num_unacceptable += 1
			print("I/O error({0}): {1}".format(e.errno, e.strerror))
		except ValueError:
			num_unacceptable += 1
			print("Conversion or value error.")
		except:
			num_unacceptable += 1
			print("Unexpected error found: ", sys.exc_info()[0])
			raise
	debugPickle(f)

#entry_src = 'data/lang-8-entries.txt'
#deleteDuplicateURLs(data_src, new_src)
runSimpleTests(new_src, PICKLE_FILE)