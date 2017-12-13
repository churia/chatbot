from nltk.tokenize import word_tokenize
import sys
reload(sys)  
sys.setdefaultencoding('Cp1252')

def tokenize(sentence):
	#print sentence
	words = word_tokenize(sentence)
	return " ".join(words)

inf = open("dialogues.txt")
outf = open("dialogues.token","w")

for l in inf:
	strs = l.rstrip().split("+++$+++")
	output = strs[0]+"+++$+++"+strs[1]+"+++$+++"+tokenize(strs[2])+"+++$+++"+tokenize(strs[3])
	outf.write(output+"\n")
inf.close()
outf.close()
	
