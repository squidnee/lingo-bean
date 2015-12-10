from sklearn import cross_validation
from sklearn.svm import SVC
import numpy as np
import random, sys, time
from utils import languages
from SetProcessing import SetProcessing
from rnn import RNN

def crossValidation(train_x, train_y, test_x, test_y):
	train_x, train_y, test_x, test_y = cross_validation.train_test_split(test_size=0.4)
	clf = SVC(kernel='linear', C=1).fit(X_train, y_train)
	scores = cross_validation.cross_val_score(clf, train_x, train_y, scoring='f1_weighted')

if __name__ == '__main__':
	s = {'fold':3, # 5 folds 0,1,2,3,4
         'lr':0.0627142536696559,
         'verbose':1,
         'decay':False, # decay on the learning rate if improvement stops
         'win':7, # number of words in the context window
         'bs':9, # number of backprop through time steps
         'nhidden':100, # number of hidden units
         'seed':345,
         'emb_dimension':100, # dimension of word embedding
         'nepochs':50,
         'vocab_size':1000}

	sp = SetProcessing()
	train = sp.convertDataToList(sp.train)
	dev = sp.convertDataToList(sp.dev)
	test = sp.convertDataToList(sp.test)

	train_x, train_y = sp.returnEntriesWithSpoken(train)
	dev_x, dev_y = sp.returnEntriesWithSpoken(dev)
	test_x, test_y = sp.returnEntriesWithSpoken(test)

	all_train_x = train_x.append([i for i in dev_x])
	all_train_y = train_y.append([j for j in dev_y])

	np.random.seed(s['seed'])
	random.seed(s['seed'])

	'''
        nh :: dimension of the hidden layer
        nc :: number of classes
        ne :: number of word embeddings in the vocabulary
        de :: dimension of the word embeddings
        cs :: word window context size 
        '''
	rnn = RNN(	nh = s['nhidden'],
				nc = len(languages()),
				ne = s['vocab_size'],
				de = s['emb_dimension'],
				cs = s['win'])

	best_f1 = -numpy.inf
	s['clr'] = s['lr']
	for e in range(s['nepochs']):
		# shuffle
		shuffle([train_lex, train_ne, train_y], s['seed'])
		s['ce'] = e
		tic = time.time()
		for i in xrange(nsentences):
			cwords = contextwin(train_lex[i], s['win'])
			words  = map(lambda x: numpy.asarray(x).astype('int32'),\
							minibatch(cwords, s['bs']))
			labels = train_y[i]
			for word_batch , label_last_word in zip(words, labels):
				rnn.train(word_batch, label_last_word, s['clr'])
				rnn.normalize()
			if s['verbose']:
				print('[learning] epoch %i >> %2.2f%%'%(e,(i+1)*100./nsentences),'completed in %.2f (sec) <<\r'%(time.time()-tic))
				sys.stdout.flush()

	#cross_validation(all_train_x, all_train_y, test_x, test_y)