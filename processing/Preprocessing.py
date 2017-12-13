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
            
