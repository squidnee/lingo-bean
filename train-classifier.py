import collections, sys, csv, itertools, pyplot, nltk, random
import numpy as np
from nltk.corpus import treebank
from datetime import datetime
from utils import readTrainingExamples, dotProduct
    '''
    Documentation for nltk can be found at 'www.nltk.org'.
    Parsing word and sentence features inspired by a tutorial on
    Recurrent Neural Networks, found at 'http://www.wildml.com/'.
    '''

data_src = 'data/testing-correction-helper.txt'

def parseSentenceFeatures(corrections):
    start_token = "SENTENCE_START"
    end_token = "SENTENCE_END"
    sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in corrections])
    sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
    return sentences

def parseWordFeatures(sentences, vocab_cap=10000):
    unknown = "UNKNOWN_WORD"
    words = [nltk.word_tokenize(sent) for sent in sentences]
    frequencies = findWordFrequencyDists(words)

    vocab = frequencies.most_common(vocab_cap-1)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown)
    word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
 
    for i, word in enumerate(words):
        words[i] = [w if w in word_to_index else unknown for w in word]

    x = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in sentences])
    y = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in sentences])
    return words, x, y

def tagWords(words, entities=False):
    tagged_words = nltk.pos_tag(words)
    entities = nltk.chunk.ne_chunk(tagged_words)
    if entities is True:
        return tagged_words, entities
    else:
        return tagged_words

def findWordFrequencyDists(words, bool_contexts=False):
    # Frequency distribution of each individual word.
    word_freq = nltk.FreqDist(itertools.chain(*words))

    # Frequency distribution mapping each context to the number
    # of times that the context was used.
    if bool_contexts is True:
        comm_contexts = nltk.common_contexts(words)
        return word_freq, comm_contexts
    else:
        return word_freq

def StochasticGradientDescent(trainingSet, featureExtractor, lossFunction=None):
    # HYPERPARAMETERS: step size (eta), number of iterations (T)
    weights = collections.defaultdict(float)
    T = 20
    for t in range(T):
        eta = float(1/float(math.sqrt(t+1)))
        for example in trainingSet:
            features = featureExtractor(example[0])
            loss = 1 - (example[1] * dotProduct(weights, features))
            for feature in features:
                weights[feature] -= eta*(-features[feature]*example[1]) if (loss >= 1) else 0
    return weights

# TODO: Have your data collection default mechanism support minibatch gradient descent!!
def MinibatchGradientDescent(trainingSet, featureExtractor, lossFunction=None):
    # HYPERPARAMETERS: step size (eta), number of iterations (T), number of batches (numBatches)
    weights = collections.defaultdict(float)
    T = 20
    numBatches = 50
    for t in range(T):
        eta = float(1/float(math.sqrt(t+1)))
        batch = random.sample(trainingSet, numBatches)
        features = featureExtractor(batch) #TODO: support batch support in your feat. extractor
        loss = sum(example * dotProduct(weights, features) for example in batch)
        for feature in features:
            weights[feature] -= eta*(-features[feature]*example[1]) if (loss >= 1) else 0
    return weights

corrections = None
sentences = parseCorrectSentences(corrections)