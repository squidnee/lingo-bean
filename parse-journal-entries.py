#!/usr/bin/python
# coding: utf8
from __future__ import unicode_literals
import sys
import collections
import requests, csv
from bs4 import BeautifulSoup, NavigableString
from CorrectionHelper import CorrectionHelper
import sqlite3 as sql
import pandas as pd

try:
    import timeoutsocket
    timeoutsocket.setDefaultSocketTimeout(10)
except ImportError:
    pass

# Creates 'cleaned' text file with all duplicate URLs removed. Can comment out once file has been created.

# def deleteDuplicateURLs(data_src, new_src):
# 	bad_suffix = ['.1/']
# 	with open(data_src, 'r') as oldfile, open(new_src, 'w') as newfile:
# 		for url in oldfile:
# 			if not any(suffix in url for suffix in bad_suffix):
# 				newfile.write(url)

# Parses learner data from each URL journal entry in the text file via BeautifulSoup.
# Returns the following information:
#	-speaking: language currently spoken by learner
#	-studying: target language being studied by learner
#	-entry: full text of the journal entry
#	-incorrect: sentences deemed incorrect by native speakers
#	-correct: corrected versions of aforementioned sentences
def mineLearnerData(url):
	resp = requests.get(url)
	raw_text = resp.text
	soup = BeautifulSoup(raw_text, 'html.parser', from_encoding="gb18030")
	for s in soup.findAll('script'):
		s.replaceWith('')
	entry = soup.select('div#body_show_ori')[0].text
	speaking = soup.select('li.speaking')[0].text
	studying = soup.select('li.studying')[0].text
	incorrect, correct = returnCorrectedSets(soup)
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

def buildSQLDatabase(new_src, c):
	with open(new_src, 'r') as f:
		count = 0
		for url in f.readlines():
			count += 1
			speaking, studying, entry, incorrect, correct = mineLearnerData(url)
			data = [speaking, studying, entry, str(incorrect), str(correct)]
			cols = ':'+', :'.join(data)
			query = "INSERT INTO languageData (speaking, studying, entry, incorrect, correct) VALUES (%s)" % cols
			c.execute(query)
			print(count)
	conn.commit()
	f.close()
	conn.close()
	print("Done!")

conn = sql.connect('language-data.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS languageData
             (speaking text, studying text, entry text, incorrect text, correct text)''')

data_src = 'data/lang-8-url-201012.txt'
new_src = 'data/lang-8-url-cleaned.txt'
#entry_src = 'data/lang-8-entries.txt'

#deleteDuplicateURLs(data_src, new_src)
buildSQLDatabase(new_src, c)