import random

""" save movie title in a dictionary, where key is movie id and value is move title
"""
def getMovieTitles(path):
    movieTitleDic = {}
    movietitleFile = open(path, 'r')
    for line in movietitleFile:
        lineList = line.split("+++$+++")
        movieTitleDic[lineList[0].rstrip()] = lineList[1].strip()
    movietitleFile.close()
    return movieTitleDic

movieTitleDic = getMovieTitles('movie_titles_metadata.txt')

"""save movie lines in a dictionary, where key is line id and value is line
"""
def getMovieLines(path):
    movieLineDic = {}
    movielinesFile = open(path, 'r')
    for line in movielinesFile:
        lineList = line.split("+++$+++")
        movieLineDic[lineList[0].rstrip()] = lineList[-1].rstrip()
    movielinesFile.close()
    return movieLineDic

movieLineDic =  getMovieLines('movie_lines.txt')

def getLineList(stringList):
    l = []
    for s in stringList:
        startIndex = s.find('L')
        ss = s[startIndex:].rstrip('\n,]\'')
        l.append(ss)
    return l

"""write dialogue to disk: movie id, movie title, utterance1, utterance 2 ..."""
def writeDialogues(outputPath, conversationPath, movieTitleDic, movieLineDic):
    outputFile = open(outputPath, 'w')
    conversationFile = open( conversationPath, 'r')
    dialoguenum = 0
    utteranceNum = 0
    for line in conversationFile:
        dialoguenum = dialoguenum + 1
        lineList = line.split("+++$+++")
        movieId = lineList[2].strip()
        movieTitle = movieTitleDic.get(movieId)
        conversationString = lineList[3].split(',')
        conversationList = getLineList(conversationString)
        conversionNum = len(conversationList)
        response = movieLineDic.get( conversationList[conversionNum - 1])
        content = ''
        for i in range(conversionNum - 1):
            utteranceNum = utteranceNum + 1
            content = content + movieLineDic[conversationList[i]]
        outputFile.write(movieId + "+++$+++" +movieTitle +"+++$+++" + content + "+++$+++" +response + '\n')
    print 'dialogue num is ' + str(dialoguenum)
    print 'utterance num is ' + str(utteranceNum)
    outputFile.close()
    conversationFile.close()
        
        

writeDialogues('dialogues.txt', 'movie_conversations.txt', movieTitleDic, movieLineDic)
            
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
        print content 
        print response
        testPath.write(content + '+++$+++' + response.strip('\n'))
        for i in range(10):
            index = random.randint(1,(lineNum - int(lineNum*0.7)))
            s =  testLines[index - 1].split("+++$+++")
            curfake = s[3].strip('\n')
            print curfake
            testPath.write('+++$+++' + curfake)
        testPath.write('\n')
    testPath.close()
    inputFile.close()


getTrainingandTest('dialogues.txt', 'training.txt', 'test.txt') 
    
