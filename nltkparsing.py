from __future__ import division, print_function, unicode_literals
import collections, sys, random
import nltk
from nltk.tokenize import MWETokenizer

'''Documentation for nltk can be found at 'www.nltk.org'.
'''

def parseSentenceFeatures(entry):
    #start_token = "SENTENCE_START"
    #end_token = "SENTENCE_END"
    sentences = nltk.sent_tokenize(entry)
    #sentences = ["%s %s %s" % (start_token, x, end_token) for x in sentences]
    return sentences

def parseWordsFromEntry(entry, vocab_cap=10000):
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
    tagged_words = nltk.pos_tag(words)
    entities = nltk.chunk.ne_chunk(tagged_words)
    if entities is True:
        return tagged_words, entities
    else:
        return tagged_words

def findWordFrequencyDists(words, bool_contexts=False):
    # Frequency distribution of each individual word.
    word_freq = nltk.FreqDist(words)

    # Frequency distribution mapping each context to the number
    # of times that the context was used.
    if bool_contexts is True:
        comm_contexts = nltk.common_contexts(words)
        return word_freq, comm_contexts
    else:
        return word_freq