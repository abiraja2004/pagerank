from collections import Counter
import math
import numpy
import nltk
from numpy import linalg
import re
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

def readTextFile(filename):
    file = open(filename, "r")
    textData = file.read()
    return textData

def filterTextCorpus(corpus, language):

    filterdCorpus = []

    for sent in corpus:
        filterdSentence = sent.lower()
        filterdSentence = re.sub('[^A-Za-z0-9 ]+', '', filterdSentence)
        filterdSentence = removeStopWords(filterdSentence, language)
        filterdSentence = stemmSentence(filterdSentence)

        if filterdSentence is "":
            continue

        filterdCorpus.append(filterdSentence)

    return filterdCorpus

def removeStopWords(sentence, language):
    stop_words = set(stopwords.words(language))
    filterdSentence = ""

    wordList = nltk.word_tokenize(sentence)

    for word in wordList:
        if word not in stop_words and len(word) > 2:
            word = word.lower()
            filterdSentence += word + " "

    return filterdSentence

def stemmSentence(sentence):
    stemmedTokens = ""
    stemmer = nltk.stem.RSLPStemmer()
    wordList = nltk.word_tokenize(sentence)
    for token in wordList:
        stemmed = stemmer.stem(token)
        stemmedTokens += stemmed + " "

    return stemmedTokens

def cosineSimilarity(vectorA, vectorB):
    dotResult = numpy.dot(vectorA, vectorB)
    n1 = linalg.norm(vectorA)
    n2 = linalg.norm(vectorB)

    if (n1 * n2) != 0:
        sim = dotResult / (n1 * n2);
    else:
        sim = 0

    return sim


def calcTfArray(termCountDoc, vocabulary):
    tfArray = []
    for vocab in vocabulary:
        tfArray.append(termCountDoc.get(vocab, 0))

    return tfArray


def calcTfIdfArray(tfArray, vocabulary, termInDocuments, documentN):
    tfdidfArray = []
    for index, tf in enumerate(tfArray):
        vocab = vocabulary[index]
        tfidfValue = tf * (1 + math.log((documentN / (termInDocuments[vocab]))))
        tfdidfArray.append(tfidfValue)

    return tfdidfArray


def normalizeTermFrequency(wordCounts):
    maxFreq = max(wordCounts.values())
    for key in wordCounts:
        wordCounts[key] /= maxFreq

    return wordCounts


def generateDocumentVector(document):
    # remove punctuation, lemmatation, stemming
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(document)

    wordCount = Counter(tokens)
    normalizedWordCount = normalizeTermFrequency(wordCount)

    return normalizedWordCount


def generateDocumentWordCountAndVocabulary(documentCorpus,  vocabularySet):
    documentWordCounts = []
    # generate vocabulary and term count per document
    for sentence in documentCorpus:
        normalizedWordCount = generateDocumentVector(sentence)
        vocabularySet.update(normalizedWordCount.keys())
        documentWordCounts.append(normalizedWordCount)

    return documentWordCounts


def generateTermCountsPerDocument(documentsWordCounts, vocabulary, termInDocuments):
    # generate term count for all sentences:
    for vocab in vocabulary:
        for dokumentVectorWordCount in documentsWordCounts:
            if dokumentVectorWordCount.get(vocab, 0) != 0:
                value = termInDocuments.get(vocab, 0)
                value = value + 1
                termInDocuments[vocab] = value


def generateTfMatrixPerDocument(documentsWordCounts, vocabulary):
    tfMatrix = []
    for counterDoc in documentsWordCounts:
        tfArray = calcTfArray(counterDoc, vocabulary)
        tfMatrix.append(tfArray)

    return tfMatrix


def generateTfIdfMatrixPerDocument(tfMatrix, vocabulary, termInDocuments, documentN):
    tfidfMatrix = {}
    for id, tfArray in enumerate(tfMatrix):
        tfdidfArray = calcTfIdfArray(tfArray, vocabulary, termInDocuments, documentN)
        tfidfMatrix[id] = tfdidfArray

    return tfidfMatrix


def simpleTfIdf(documentCorpus):
    vocabularySet = set()
    termInDocuments = {}

    # count of sentences (used in idf calculation)
    documentN = len(documentCorpus)

    documentsWordCounts = generateDocumentWordCountAndVocabulary(documentCorpus,  vocabularySet)
    vocabulary = list(vocabularySet)

    generateTermCountsPerDocument(documentsWordCounts, vocabulary, termInDocuments)

    tfMatrix = generateTfMatrixPerDocument(documentsWordCounts, vocabulary)

    # generate tfidf array for each sentence:
    tfidfMatrix = generateTfIdfMatrixPerDocument(tfMatrix, vocabulary, termInDocuments, documentN)
    return tfidfMatrix

def compareDifferenceDict(dict1, dict2):
    diffDict = {}
    for key in dict1:
        diffDict[key] = abs(dict1[key] - dict2[key])

    return diffDict

def checkConvergence(diffDict, epsilon):
    for value in diffDict.values():
        if value > epsilon:
            return False

    return True

def buildSimilarityGraph(tfidfMatrix, threshold):

    graph = {}
    for key in tfidfMatrix.keys():
        graph[key] = []

    for key in tfidfMatrix.keys():
        vectorA = tfidfMatrix[key]
        for key2 in tfidfMatrix.keys():

            #ignore self similarity
            if key == key2:
                continue

            vectorB = tfidfMatrix[key2]
            sim = cosineSimilarity(vectorA, vectorB)
            if sim > threshold:
                graph[key].append(key2)

    return graph

def pageRank(linkGraphDict, dampingFactor, numberOfIterations):
    pageRankDict = {}
    allIterations = []
    allDiffIterations = []

    N = len(linkGraphDict.keys())

    #init pageRankvector
    for node in linkGraphDict.keys():
        pageRankDict[node] = 1.0/ N

    for iteration in range(numberOfIterations):
        pageRankCurIteration = {}

        for node, values in linkGraphDict.items():

            linkSum = 0
            for linkFromNode in values:
                linkSum = linkSum +  pageRankDict[linkFromNode] / len(linkGraphDict[linkFromNode])

            rank = (dampingFactor / N) + (1 - dampingFactor) * linkSum
            pageRankCurIteration[node] = rank

        diffDict = compareDifferenceDict(pageRankDict, pageRankCurIteration)
        pageRankDict = pageRankCurIteration
        allIterations.append(pageRankCurIteration)
        allDiffIterations.append(diffDict)

        isConverged = checkConvergence(diffDict, 0.000000001)
        if isConverged:
            #print("Number of iterations to converge: " + str(iteration))
            return pageRankDict

    return pageRankDict


pathToTextFile = "trump.txt"
text = readTextFile(pathToTextFile)
corpus = sent_tokenize(text)
filteredCorpus = filterTextCorpus(corpus, "english")

tfidfMatrix = simpleTfIdf(filteredCorpus)
simGraph = buildSimilarityGraph(tfidfMatrix, 0.12)
pageRankRes = pageRank(simGraph, 0.15, 50)

sortedList = sorted(pageRankRes.items(), key=lambda x:x[1],  reverse=True)
top5Results = sortedList[:5]
top5Results = sorted(top5Results, key=lambda x:x[0])

for result in top5Results:
    print(corpus[result[0]])
    print()