from __future__ import print_function
import csv, random, pickle
from collections import defaultdict
'''
    Some of these functions were barely modified from the utils.py script
    in the original CS 221 'Sentiment' assignment.
    '''
TRAIN_PATH = 'data/all-data.pickle'
DEV_PATH = 'data/all-data-3.pickle'
TEST_PATH = 'data/all-data-2.pickle' # TODO: UPDATE THESE DATA SETS

def asciistrip(string):
    '''Strips the string of that pesky ascii code that rfind can't find.'''
    return string.encode('utf-8').decode('ascii', 'ignore').strip()

def filepaths():
    '''Returns the file paths of each dataset for easy access.'''
    return TRAIN_PATH, DEV_PATH, TEST_PATH

def languages():
    '''Returns a list of the native languages being considered in this project.'''
    return ['English', 'Spanish', 'French', 'Korean', 'Japanese', 'Mandarin']

def unicodeReader(data):
    '''Reads unicode within a CSV file.'''
	reader = csv.reader(data)
	for row in reader:
		yield [cell.encode('utf-8') for cell in row]

def unpickleFile(pickle_path):
    '''Unpickles a file.'''
    f = open(pickle_path, 'rb')
    my_list_of_dicts = pickle.load(f)
    f.close()
    return my_list_of_dicts

def writeToTextfile(path, output):
    '''Writes the specified output to a text file.'''
    with open(txtpath) as f:
        f.write(output)
    f.close()

def returnDatasets():
    '''Returns each dataset for easy access.'''
    train = unpickleFile(TRAIN_PATH)
    dev = unpickleFile(DEV_PATH)
    test = unpickleFile(TEST_PATH)
    return train, dev, test

def retrieveDatasetsWithStats():
    '''Returns each dataset coupled with how many entries it contains and what %
       of all considered entries it contains.'''
    train_dicts, dev_dicts, test_dicts = returnDatasets()
    train_n = len(train_dicts.keys())
    dev_n = len(dev_dicts.keys())
    test_n = len(test_dicts.keys())

    N = train_n + dev_n + test_n
    train_perc = (train_n / N) * 100
    dev_perc = (dev_n / N) * 100
    test_perc = (test_n / N) * 100
    datasets = (train_dicts, dev_dicts, test_dicts)
    testStats = [("Training N: " + str(train_n), "Training %: " + str(train_perc)), \
                ("Development N: " + str(dev_n), "Development %: " + str(dev_perc)), \
                ("Test N: " + str(test_n), "Test %: " + str(test_perc))]
    return datasets, testStats

def makeLangPrefixMapping():
    '''Returns a map of languages to their prefixes, as used by the Google API and many
       language parsers.'''
    lang_mapping = {'German': 'de', 'Spanish': 'es', 'French': 'fr', 'Japanese': 'ja', \
     'English': 'en', 'Korean': 'ko', 'Mandarin': 'zh-CN'}
    return lang_mapping

def makePrefixLangMapping():
    '''Switches the values and the keys for the language-prefix map above.'''
    lang_mapping = makeLangPrefixMapping()
    pref_mapping = dict(zip(lang_mapping.values(),lang_mapping.keys()))
    return pref_mapping

def dotProduct(d1, d2):
    '''Computes the dot product on a vector. (To be used with SGD)'''
    if len(d1) < len(d2):
    	return dotProduct(d2, d1)
    else: 
    	return sum(d1.get(f, 0) * v for f, v in d2.items())

def evaluateClassifier(data, classifier):
    '''Evaluates a predictor/classifier.'''
    error = 0
    for x, y in data:
        if classifier(x) != y: error += 1
    return 1.0 * error / len(data)