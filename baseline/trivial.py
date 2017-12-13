import random
import sys
import numpy as np
from nltk import corpus
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize

def randomPick(rangemax=10):
	result = range(0,rangemax)
	random.shuffle(result) 
	return result

class TFIDFPredictor:
	def __init__(self,ngram=2,stop=False):
		if stop:	
			stopwords = corpus.stopwords.words('english')
		else:
			stopwords = None
		self.vectorizer = TfidfVectorizer(decode_error='ignore',encoding='Cp1252',stop_words=stopwords, max_features=100000, ngram_range=(1,ngram))

	def train(self,document):
		self.vectorizer.fit(document)

	def predict(self,content,candidates):
		content_vector = self.vectorizer.transform([content])
		candidates_vector = self.vectorizer.transform(candidates)
		content_vector = normalize(content_vector,norm='l2', axis=1)
		candidates_vector = normalize(candidates_vector, norm='l2', axis=1)
		result = (content_vector*(candidates_vector.T)).todense()
		result = np.asarray(result).flatten()
		flag = 0
		if (np.count_nonzero(result)) == 0:	
			flag = 1
			return flag, randomPick()
		return flag,np.argsort(result,axis=0)[::-1]
	
trainfile = sys.argv[1]
testfile = sys.argv[2]

#training
doc = []
for fil in [trainfile,testfile]:
	with open(fil) as f:
		f.readline()
		for l in f.readlines():
			doc.append(" ".join(l.strip().split("+++$+++")))
tfidf = TFIDFPredictor()
tfidf.train(doc)

#testing
krange = [1,2,5]
with open(testfile) as f:
	f.readline()
	lines = f.readlines()
	total = 0.0
	random_correct = {}
	tfidf_correct = {}
	for k in krange:
		random_correct[k] = 0.0
		tfidf_correct[k] = 0.0
	count = 0
	for l in lines:
		total += 1
		strs = l.strip().split("+++$+++")
		content = strs[0]
		response = strs[1]
		candidates = strs[1:]

		random_answer = randomPick()
		flag, tfidf_answer = tfidf.predict(content,candidates)
		if flag == 1:
			count +=1
		for k in krange:
			if 0 in random_answer[0:k]:
				random_correct[k] += 1	
			if 0 in tfidf_answer[0:k]:
				tfidf_correct[k] += 1
	for k in krange:
		print "Random Pick: R@("+str(k)+", 11): "+str(random_correct[k]/total)
	for k in krange:
		print "TFIDF model: R@("+str(k)+", 11): "+str(tfidf_correct[k]/total)
	print "No similarity between content and response using:", count/total 
