from __future__ import division, print_function, unicode_literals
import collections, sys, random
import nltk
from nltk.tokenize import MWETokenizer

'''Further documentation for nltk can be found at 'www.nltk.org'.
'''

def parseSentenceFeatures(entry):
    '''Tokenizes an entry into a list of sentences.'''
    return nltk.sent_tokenize(entry)

def parseWordsFromEntry(entry, vocab_cap=10000):
    '''Tokenizes an entry into a list of words.'''
    '''Calculates their indeces relative to their frequencies.'''
    unknown = "UNKNOWN_WORD"
    tokenizer = MWETokenizer()
    words = entry.split()
    #words = tokenizer.tokenize(entry.split())
    frequencies = findWordFrequencyDists(words)
    vocab = frequencies.most_common(vocab_cap-1)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown)
    word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
    return word_to_index

def parseWordsFromSentences(sentences, vocab_cap=10000):
    '''Tokenizes a list of sentences into a list of words.'''
    '''Calculates their indeces relative to their frequencies.'''
    unknown = "UNKNOWN_WORD"
    words = []
    for sent in sentences:
        words += sent.split()
    frequencies = findWordFrequencyDists(words)
    vocab = frequencies.most_common(vocab_cap-1)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown)
    word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
    return word_to_index

def tagWords(words, entities=False):
    '''Tags the POS of each word.'''
    tagged_words = nltk.pos_tag(words)
    entities = nltk.chunk.ne_chunk(tagged_words)
    if entities is True:
        return tagged_words, entities
    else:
        return tagged_words

def findWordFrequencyDists(words, bool_contexts=False):
    '''Finds the frequency distribution of each word.'''
    return nltk.FreqDist(words)