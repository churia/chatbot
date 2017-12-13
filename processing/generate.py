import random

"""use 70% percent of data as training and 30 percent as test data"""
def getTrainingandTest(inputPath, traningOutput, testOutput):
    inputFile = open(inputPath, 'r')
    traningPath = open(traningOutput, 'w')
    traningPath.write('content' + "+++$+++" + 'response\n')
    testPath = open(testOutput, 'w')
    testPath.write('content' + "+++$+++" + 'true response' + '+++$+++')
    for i in range(10):
        if i != 9:
            testPath.write('fake' + str(i + 1) + '+++$+++')
        else:
            testPath.write('fake' + str(i + 1) + '\n')
    lines = inputFile.readlines()
    random.shuffle(lines)
    lineNum = len(lines)
    traininglines = lines[:int(lineNum*0.7)]
    testLines = lines[int(lineNum * 0.7):]
    for line in traininglines: 
        lineList = line.split("+++$+++")
        content = lineList[2]
        response = lineList[3]
        traningPath.write(content + '+++$+++' + response)
    traningPath.close()
    for line in testLines:
        lineList = line.split("+++$+++")
        content = lineList[2]
        response = lineList[3]
#        print content 
#        print response
        testPath.write(content + '+++$+++' + response.strip('\n'))
        for i in range(10):
            index = random.randint(1,(lineNum - int(lineNum*0.7)))
            s =  testLines[index - 1].split("+++$+++")
            curfake = s[3].strip('\n')
#            print curfake
            testPath.write('+++$+++' + curfake)
        testPath.write('\n')
    testPath.close()
    inputFile.close()


#getTrainingandTest('dialogues.txt', 'training.txt', 'test.txt') 
getTrainingandTest('dialogues.token', 'training.token', 'test.token') 
    
