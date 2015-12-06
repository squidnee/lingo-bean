from __future__ import print_function
import random, collections
from utils import dotProduct, readTrainingExamples
from giveawayFeatures import featureExtractor

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

def KNearestNeighbors():
    pass #TODO or import from library