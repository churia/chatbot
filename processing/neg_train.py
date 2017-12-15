import sys
import random

trainfile = sys.argv[1]
outfile = sys.argv[2]
lines=[]
with open(trainfile) as f:
	title = f.readline()
	lines = f.readlines()


split = "+++$+++"
length = len(lines)
count=0
with open(outfile,'w') as f:
	f.write(title.replace("response","fake_response"))
	for l in lines:
		strs=l.rstrip().split(split)
		content = strs[0]
		index=random.randint(0,length-1)
		while(index==count):
			index=random.randint(0,length-1)
		response = lines[index].rstrip().split(split)[1]
		f.write(content+split+response+"\n")
		count += 1
