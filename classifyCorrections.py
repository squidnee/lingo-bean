import time
from wordvectors import *
#from gensim.models import Word2Vec
from textblob import TextBlob
from sklearn.cluster import KMeans

def runKMeans(vectors, K, model=None): #TODO
	start = time.time()
	word_vecs = vectors # TODO
	cluster_total = K
	kmeans_alg = KMeans(n_clusters=cluster_total)
	assignments = kmeans_alg.fit_predict( word_vecs )
	end = time.time()
	timeTaken = end - start
	print("Time taken for KMeans: " + timeTaken, + " seconds")
	classif_clusters = dict(zip( model.index2word, assignments ))
	for cluster in xrange(cluster_total):
		words = []
		if i in xrange(len(classif_clusters.values())):
			if classif_clusters.values()[i] is cluster:
			 words.append(classif_clusters.keys()[i])
		print(words)

def collectSpellingErrors(sentences):
	spelling_errors_counter = 0
	misspelled_words = []
	for sentence in sentence:
		blob = TextBlob(sentence)
		correction = blob.correct()
		if correction != sentence:
			for word in sentence.split():
				if word not in correction:
				misspelled_words.append(word)
				spelling_errors_counter += 1
	print("There were " + str(spelling_errors_counter) + " spelling errors in this entry.")
	return misspelled_words

if __name__ == '__main__':
	model = None # TODO
	clusters = 6
	vectors = None
	runKMeans(vectors, clusters, model)