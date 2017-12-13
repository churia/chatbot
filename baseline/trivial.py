import random
import sys
import numpy as np
from nltk import corpus
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def randomPick(rangemax=10):
	return random.randint(0,rangemax)

class TFIDFPredictor:
	def __init__(self,ngram=1,stop=False):
		if stop:	
			stopwords = corpus.stopwords.words('english')
		else:
			stopwords = 'english'
		self.vectorizer = TfidfVectorizer(decode_error='ignore',encoding='Cp1252',stop_words=stopwords, max_features=10000, ngram_range=(1,ngram))

	def train(self,document):
		self.vectorizer.fit(document)

	def predict(self,content,candidates):
		content_vector = self.vectorizer.transform([content])
		candidates_vector = self.vectorizer.transform(candidates)
		result = (content_vector*(candidates_vector.T)).todense()
		result = np.asarray(result).flatten()
		return np.argmax(result,axis=0)
	
trainfile = sys.argv[1]
testfile = sys.argv[2]

#training
doc = []
with open(trainfile) as f:
	f.readline()
	for l in f.readlines():
		doc.append(" ".join(l.strip().split("+++$+++")))
tfidf = TFIDFPredictor()
tfidf.train(doc)

#testing
with open(testfile) as f:
	f.readline()
	lines = f.readlines()
	total = 0.0
	random_correct = 0.0
	tfidf_correct = 0.0
	for l in lines:
		total += 1
		strs = l.strip().split("+++$+++")
		content = strs[0]
		response = strs[1]
		candidates = strs[1:]

		random_answer = randomPick()
		tfidf_answer = tfidf.predict(content,candidates)
		if random_answer == 0:
			random_correct += 1 
		if tfidf_answer == 0:
			tfidf_correct += 1 
	print "Accuracy of Random Pick: "+str(random_correct/total)
	print "Accuracy of TFIDF model: "+str(tfidf_correct/total)
