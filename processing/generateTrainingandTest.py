
# coding: utf-8

# In[16]:

import random
from random import *
from random import shuffle
def generateTrainingTestandValidation(orginPath):
    lines = []
    with open(orginPath, 'r') as f:
        lines = f.readlines()
    shuffle(lines)
    num = len(lines)
    trainingLines = lines[:int(num * 0.3)]
    validLines = lines[(int)(num * 0.3): (int)(num * 0.3 + num * 0.1)]
    testLines = lines[(int)(num * 0.3 + num * 0.1) :]
    with open('training.txt', 'w') as f:
        trainingnum = len(trainingLines)
        f.write('context, response, label\n')
        count = 0
        for line in trainingLines:
            linelist = line.split('+++$+++')
            content = linelist[2]
            trueResponse = linelist[3].strip()
            count += 1
            p = random()
            if p >= 0.5:
                f.write(content.strip() + '+++$+++' + trueResponse + '+++$+++'+'1\n')
            else:
                index = uniform(1, trainingnum)
                index =  int(index)
                while index == count:
                    index = uniform(1, trainingnum)
                f.write(content.strip() + '+++$+++' + trainingLines[(int)(index - 1)].split('+++$+++')[3].strip()+'+++$+++' + '0\n')
    with open('test.txt', 'w') as f:
        f.write('content' + '+++$+++' + 'response' + '+++$+++')
        for i in range(10):
            if i != 9:
                f.write('fake' + str(i + 1) + '+++$+++')
            else:
                f.write('fake' + str(i + 1) + '\n')
        lineNum = len(testLines)
        count = 0
        for line in testLines: 
            count += 1
            lineList = line.split("+++$+++")
            content = lineList[2]
            response = lineList[3]
            f.write(content.strip() + '+++$+++' + response.strip('\n'))
            for i in range(10):
                index = uniform(1, lineNum)
                index = int(index)
                while index == count:
                    index = uniform(1, lineNum)
                    index = int(index)
                s =  testLines[index - 1].split("+++$+++")
                curfake = s[3].strip('\n')
                f.write('+++$+++' + curfake)
            f.write('\n')
    with open('validation.txt', 'w') as f:
        f.write('content' + '+++$+++' + 'response' + '+++$+++')
        for i in range(10):
            if i != 9:
                f.write('fake' + str(i + 1) + '+++$+++')
            else:
                f.write('fake' + str(i + 1) + '\n')
        lineNum = len(validLines)
        count = 0
        for line in validLines: 
            count += 1
            lineList = line.split("+++$+++")
            content = lineList[2]
            response = lineList[3]
            f.write(content.strip() + '+++$+++' + response.strip('\n'))
            for i in range(10):
                index = uniform(1, lineNum)
                index = int(index)
                while index == count:
                    index = uniform(1, lineNum)
                    index = int(index)
                s =  validLines[index - 1].split("+++$+++")
                curfake = s[3].strip('\n')
                f.write('+++$+++' + curfake)
            f.write('\n')
    
        

    
generateTrainingTestandValidation('dialogues.txt')


# In[ ]:



