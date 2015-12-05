#!/usr/bin/python
# coding: utf8
from __future__ import unicode_literals, print_function
import sys
import collections
import requests, pickle
from bs4 import BeautifulSoup
from DataCleaner import DataCleaner
from utils import asciistrip as astrip
from utils import unpickleFile

try:
    import timeoutsocket
    timeoutsocket.setDefaultSocketTimeout(10)
except ImportError:
    pass

TRAIN_PICKLE_PATH =  "data/TrainSet.pickle"
DEV_PICKLE_PATH = "data/DevSet.pickle"
TEST_PICKLE_PATH = "data/TestSet.pickle"

urlfile = 'data/lang-8-url-cleaned-2.txt'

POSSIBLE_LANGUAGES = ['English', 'Spanish', 'French', 'Korean', 'Japanese', 'Mandarin']

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
	safeToUse = True
	for s in soup.findAll('script'):
		s.replaceWith('')
	try:
		entry = soup.select('div#body_show_ori')[0].text
	except IndexError:
		entry = 'Null'
		safeToUse = False
	try:
		speaking = soup.select('li.speaking')[0].text
	except IndexError:
		speaking = 'Null'
		safeToUse = False
	try:
		studying = soup.select('li.studying')[0].text
	except IndexError:
		studying = 'Null'
		safeToUse = False
	try:
		incorrect, correct = returnCorrectedSets(soup)
	except IndexError:
		incorrect = 'Null'; correct = 'Null'
		safeToUse = False
	if not safeToUse: continue
	else return speaking, studying, entry, incorrect, correct

# Returns a set of incorrect sentences and their corrected forms, as deemed by native speakers.
# TODO: pair incorrect sentences with corrected counterparts
def returnCorrectedSets(soup):
		wrongSet = set(); correctSet = set()
		for incorrect in soup.select('div.correction_box ul.correction_field li.incorrect'):
			wrongSet.add(incorrect.text)
		for correct in soup.select('div.correction_box ul.correction_field li.corrected.correct'):
			correctSet.add(correct)
		return wrongSet, correctSet

def createPickledDatasets(list_of_dicts, pickle_file=PICKLE_FILE):
	output = open(pickle_file, 'wb')
	pickle.dump(list_of_dicts, output)
	output.close()

def collectTrainData(urlfile, pickle_file=TRAIN_PICKLE_PATH):
	# Going to have 60,000 total samples.
	f = open(urlfile, 'r')
	total_in_train = 0
	num_unacceptable = 0
	total_learner_data = dict()
	for index, url in enumerate(f.readlines()):
		if index <= 5000 or total_in_train > 60000: continue
		try:
			speaking, studying, entry, incorrect, correct = mineLearnerData(url)
			dc = DataCleaner(correct, speaking, studying)
			corrections = dc.cleanCorrections()
			studying = dc.astrip(studying).split()[0]
			if studying not in POSSIBLE_LANGUAGES: continue
			if speaking not in POSSIBLE_LANGUAGES: continue
			else: 
				learner_data = {'Speaking': speaking, 'Studying': studying, 'Entry': entry, 'Incorrect': incorrect, 'Corrections': corrections}
				total_learner_data[index] = learner_data
				createPickledDatasets(total_learner_data, pickle_file)
				total_in_train = len(total_leaner_data.keys())
				print('Total found: ' + str(index), ' Number in set: ' + str(total_in_train))
		except IOError as e:
			num_unacceptable += 1
			print("I/O error({0}): {1}".format(e.errno, e.strerror))
		except ValueError:
			num_unacceptable += 1
			print("Conversion or value error.")
		except:
			num_unacceptable += 1
			print("Unexpected error found: ", sys.exc_info()[0])
			continue

def collectDevData(urlfile, pickle_file=DEV_PICKLE_PATH):
	# Going to have 20,000 total samples.
	f = open(urlfile, 'r')
	total_in_dev = 0
	num_unacceptable = 0
	total_learner_data = dict()
	for index, url in enumerate(f.readlines()):
		if 260000 <= index <= 200000: continue
		try:
			speaking, studying, entry, incorrect, correct = mineLearnerData(url)
			dc = DataCleaner(correct, speaking, studying)
			corrections = dc.cleanCorrections()
			studying = dc.astrip(studying).split()[0]
			if studying not in POSSIBLE_LANGUAGES: continue
			if speaking not in POSSIBLE_LANGUAGES: continue
			else: 
				learner_data = {'Speaking': speaking, 'Studying': studying, 'Entry': entry, 'Incorrect': incorrect, 'Corrections': corrections}
				total_learner_data[index] = learner_data
				createPickledDatasets(total_learner_data, pickle_file)
				total_in_dev = len(total_leaner_data.keys())
				print('Total found: ' + str(index), ' Number in set: ' + str(total_in_dev))
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

def collectTestData(urlfile, pickle_file=TEST_PICKLE_PATH):
	# Going to have 20,000 total samples.
	f = open(urlfile, 'r')
	total_in_test = 0
	num_unacceptable = 0
	total_learner_data = dict()
	for index, url in enumerate(f.readlines()):
		if 260000 <= index <= 200000: continue
		try:
			speaking, studying, entry, incorrect, correct = mineLearnerData(url)
			dc = DataCleaner(correct, speaking, studying)
			corrections = dc.cleanCorrections()
			studying = dc.astrip(studying).split()[0]
			if studying not in POSSIBLE_LANGUAGES: continue
			if speaking not in POSSIBLE_LANGUAGES: continue
			else: 
				learner_data = {'Speaking': speaking, 'Studying': studying, 'Entry': entry, 'Incorrect': incorrect, 'Corrections': corrections}
				total_learner_data[index] = learner_data
				createPickledDatasets(total_learner_data, pickle_file)
				total_in_test = len(total_leaner_data.keys())
				print('Total found: ' + str(index), ' Number in set: ' + str(total_in_test))
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