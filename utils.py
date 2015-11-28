import csv
	'''
    Some of these functions were barely modified from the utils.py script
    in the original CS 221 'Sentiment' assignment.
    '''

def unicodeReader(data):
	reader = csv.reader(data)
	for row in reader:
		yield [cell.encode('utf-8') for cell in row]

def dotProduct(d1, d2):
	"""
    @param dict d1: a feature vector represented by a mapping from a feature (string) to a weight (float).
    @param dict d2: same as d1
    @return float: the dot product between d1 and d2
    """
    if len(d1) < len(d2):
        return dotProduct(d2, d1)
    else:
        return sum(d1.get(f, 0) * v for f, v in d2.items())

def readTrainingExamples(path):
    '''
    Reads a set of training examples.
    Format of each line:
    <output label (language)> <input entry>
    '''
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