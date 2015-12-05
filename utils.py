from __future__ import print_function
import csv
import pickle
#from gensim.models.word2vec import Word2Vec
#import pattern.text
'''
    Some of these functions were barely modified from the utils.py script
    in the original CS 221 'Sentiment' assignment.
    '''
TRAIN_PATH = 'data/all-data.pickle'
DEV_PATH = 'data/all-data-3.pickle'
TEST_PATH = 'data/all-data-2.pickle'

def filepaths():
    return TRAIN_PATH, DEV_PATH, TEST_PATH

def unicodeReader(data):
	reader = csv.reader(data)
	for row in reader:
		yield [cell.encode('utf-8') for cell in row]

def unpickleFile(pickle_path):
    f = open(pickle_path, 'rb')
    my_list_of_dicts = pickle.load(f)
    f.close()
    return my_list_of_dicts

def retrieveDatasetsWithStats():
    train_dicts = unpickleFile(TRAIN_PATH)
    train_n = len(train_dicts.keys())
    dev_dicts = unpickleFile(DEV_PATH)
    dev_n = len(dev_dicts.keys())
    test_dicts = unpickleFile(TEST_PATH)
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

def retrieveEntriesWithLabels(train_use=False, dev_use=False, test_use=True):
    datasets, stats = retrieveDatasetsWithStats()
    train, dev, test = datasets
    vals = []
    if train_use:
        train_pairs = dict(map(lambda x: (x['Entry'], x['Speaking']), train.values()))
        vals.append(train_pairs)
    if dev_use:
        dev_pairs = dict(map(lambda x: (x['Entry'], x['Speaking']), dev.values()))
        vals.append(dev_pairs)
    if test_use:
        test_pairs = dict(map(lambda x: (x['Entry'], x['Speaking']), test.values()))
        vals.append(test_pairs)
    return vals

def dotProduct(d1, d2):
#     @param dict d1: a feature vector represented by a mapping from a feature (string) to a weight (float).
#     @param dict d2: same as d1
#     @return float: the dot product between d1 and d2
    if len(d1) < len(d2):
    	return dotProduct(d2, d1)
    else: 
    	return sum(d1.get(f, 0) * v for f, v in d2.items())

def readTrainingExamples(path):
#     Reads a set of training examples.
#     Format of each line:
#     <output label (language)> <input entry>
    examples = []
    for line in open(path):
        y, x = line.split(' ', 1)
        examples.append((x.strip(), str(y.encode('utf-8'))))
    print('Read %d examples from %s' % (len(examples), path))
    return examples

def evaluateClassifier(data, classifier):
    error = 0
    for x, y in data:
        if classifier(x) != y: error += 1
    return 1.0 * error / len(data)