from collections import Counter
import math
import numpy
import glob
import re
import nltk
from numpy import linalg
from enum import Enum
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords


class WeightMethod(Enum):
    UNIFORM = 1
    TFIDF = 2


class PriorMethod(Enum):
    UNIFORM = 1
    POSITION = 2
    TFIDF = 3
    BAYES = 4


def readAllTextFiles(pathToFiles):
    files = glob.glob(pathToFiles + "/*.txt")
    documentTextList = []

    for file in files:
        f = open(file)
        text = f.read()
        text = text.lower()

        # remove newline :
        cText = re.sub('[a-z]\n', '. ', text)
        documentTextList.append(cText)
        f.close()

    return documentTextList

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

def generateDocumentVector(document, normalize = False):
    tokenizer = RegexpTokenizer(r'\w+')

    stop_words = set(stopwords.words("portuguese"))
    tokens = tokenizer.tokenize(document)

    wordCount = Counter(tokens)

    if normalize == True:
        wordCount = normalizeTermFrequency(wordCount)

    return wordCount

def generateDocumentWordCountAndVocabulary(documentCorpusDict, vocabularySet):
    documentWordCounts = {}
    # generate vocabulary and term count per document
    for id, sentence in documentCorpusDict.items():
        normalizedWordCount = generateDocumentVector(sentence, True)
        #normalizedWordCount = generateDocumentVectorImproved(sentence, True)
        vocabularySet.update(normalizedWordCount.keys())
        documentWordCounts[id] = normalizedWordCount

    return documentWordCounts


def generateTermCountsPerDocument(documentsWordCountsDict, vocabulary, termInDocuments):
    # generate term count for all sentences:
    for vocab in vocabulary:
        for dokumentVectorWordCount in documentsWordCountsDict.values():
            if dokumentVectorWordCount.get(vocab, 0) != 0:
                value = termInDocuments.get(vocab, 0)
                value = value + 1
                termInDocuments[vocab] = value


def generateTfMatrixPerDocument(wordCountDict, vocabulary):
    tfMatrix = {}
    for id, counterDoc in wordCountDict.items():
        tfArray = calcTfArray(counterDoc, vocabulary)
        tfMatrix[id] = tfArray

    return tfMatrix


def generateTfIdfMatrixPerDocument(tfMatrix, vocabulary, termInDocuments, documentN):
    tfidfMatrix = {}
    for id, tfArray in tfMatrix.items():
        tfdidfArray = calcTfIdfArray(tfArray, vocabulary, termInDocuments, documentN)
        tfidfMatrix[id] = tfdidfArray

    return tfidfMatrix


def simpleTfIdf(documentCorpus):
    vocabularySet = set()
    termInDocuments = {}

    # count of sentences (used in idf calculation)
    documentN = len(documentCorpus.values())

    documentsWordCounts = generateDocumentWordCountAndVocabulary(documentCorpus, vocabularySet)
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

def generateTfIdfArray(document, vocabulary, globalWordCount, documentN):
    normalizedWordCount = generateDocumentVector(document)
    tfArray = calcTfArray(normalizedWordCount, vocabulary)
    docTfIdfArray = calcTfIdfArray(tfArray, vocabulary, globalWordCount, documentN)
    return docTfIdfArray

def compareSimilarity(docTfIdfArray, tfidfMatrix):
    simResults = {}
    for idx, tfidfArray in tfidfMatrix.items():
        sim = cosineSimilarity(docTfIdfArray, tfidfArray)
        simResults[idx] = sim

    return simResults


def buildSimilarityGraph(tfidfMatrix, threshold, isWeighted=True):
    graph = {}
    for key in tfidfMatrix.keys():
        graph[key] = []

    for key in tfidfMatrix.keys():
        vectorA = tfidfMatrix[key]
        for key2 in tfidfMatrix.keys():

            # ignore self similarity
            if key == key2:
                continue

            vectorB = tfidfMatrix[key2]
            sim = cosineSimilarity(vectorA, vectorB)
            if sim > threshold:
                if isWeighted:
                    graph[key].append((key2,sim))
                else:
                    graph[key].append((key2, 1.0))

    return graph


def pageRank(linkGraphDict, prior, dampingFactor, numberOfIterations):
    pageRankDict = {}
    allIterations = []
    allDiffIterations = []

    N = len(linkGraphDict.keys())

    # init pageRankvector
    for node in linkGraphDict.keys():
        pageRankDict[node] = 1.0 / N

    for iteration in range(numberOfIterations):
        pageRankCurIteration = {}

        for node, edges in linkGraphDict.items():

            linkSum = 0
            for edge in edges:
                edgeSum = 0
                for edge2 in linkGraphDict[edge[0]]:
                    edgeSum = edgeSum + edge2[1]

                linkSum = linkSum + ((pageRankDict[edge[0]] * edge[1]) / edgeSum)

            priorProbability = (prior[node]/sum(prior.values()))
            rank = dampingFactor * priorProbability + (1 - dampingFactor) * linkSum
            pageRankCurIteration[node] = rank

        diffDict = compareDifferenceDict(pageRankDict, pageRankCurIteration)
        pageRankDict = pageRankCurIteration
        allIterations.append(pageRankCurIteration)
        allDiffIterations.append(diffDict)

        isConverged = checkConvergence(diffDict, 0.00000000001)
        if isConverged:
            #print("Number of iterations to converge: " + str(iteration))
            return pageRankDict

    #print("iteration threshhold reached")
    return pageRankDict


def generatePositionPrior(corpusDict):
    prior = {}

    N = len(corpusDict.values())
    pos = 1
    for id in corpusDict.keys():
        prior[id] = N / (pos + 50)
        pos += 1

    return prior

def generateUniformPrior(corpusDict):
    prior = {}

    for id in corpusDict.keys():
        prior[id] = 1

    return prior

def generateBayesPrior(documentCorpusDict):
    prior = {}

    docCount = len(documentCorpusDict.values())
    documentTfMatrix = {}

    for id, document in documentCorpusDict.items():
        documentTf = generateDocumentVector(document)
        documentTfMatrix[id] = documentTf

    queryText = ""
    for doc in documentCorpusDict.values():
        queryText += doc + " "

    queryTf = generateDocumentVector(queryText)

    for id, documentTf in documentTfMatrix.items():
        probQuery = 0
        docLength = sum(documentTf.values())

        for queryTerm, queryTF in queryTf.items():
            docTF = documentTf.get(queryTerm,0)
            probDoc = math.pow(((docTF + 1) / (docLength+1)), queryTF)
            probQuery = probQuery + math.log(1/docCount) + math.log(probDoc)

        prior[id] = probQuery

    return prior


def generateTfIdfPrior(corpusDict):
    prior = {}

    vocabularySet = set()
    termInDocuments = {}

    queryText = ""
    for doc in corpusDict.values():
        queryText += doc + " "

    # count of sentences (used in idf calculation)
    documentN = len(corpusDict.values())
    wordCountDict = generateDocumentWordCountAndVocabulary(corpusDict, vocabularySet)
    vocabulary = list(vocabularySet)

    generateTermCountsPerDocument(wordCountDict, vocabulary, termInDocuments)
    tfMatrix = generateTfMatrixPerDocument(wordCountDict, vocabulary)

    # generate tfidf array for each sentence:
    tfidfMatrix = generateTfIdfMatrixPerDocument(tfMatrix, vocabulary, termInDocuments, documentN)

    queryTfIdfArray = generateTfIdfArray(queryText, vocabulary, termInDocuments, documentN)
    prior = compareSimilarity(queryTfIdfArray, tfidfMatrix)

    return prior


def evaluateResults(relevantDocument, retrievedDocuments):
    correctFetchedDocs = 0
    averagePrecision = 0
    relevantDocuments = sent_tokenize(relevantDocument)

    for count,retrievedDoc in enumerate(retrievedDocuments):
        if retrievedDoc in relevantDocuments:
            correctFetchedDocs += 1
            averagePrecision += correctFetchedDocs / (count + 1)

    averagePrecision = averagePrecision / len(relevantDocuments)
    precision = correctFetchedDocs / len(retrievedDocuments)
    recall = correctFetchedDocs / len(relevantDocuments)

    if (precision + recall) != 0:
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return precision, recall, f1, averagePrecision


def printEvaluationSummary(evaluationSummary, methodName):
    map = 0
    avgPrecision = 0
    avgRecall = 0
    avgF1 = 0
    for evaluatedQuery in evaluationSummary:
        map += evaluatedQuery[3]
        avgF1 += evaluatedQuery[2]
        avgRecall += evaluatedQuery[1]
        avgPrecision += evaluatedQuery[0]

    map = map / len(evaluationSummary)
    avgF1 = avgF1 / len(evaluationSummary)
    avgRecall = avgRecall / len(evaluationSummary)
    avgPrecision = avgPrecision / len(evaluationSummary)

    print(methodName)
    print("MAP: " + str(map))
    print("Avg precision: " + str(avgPrecision))
    print("Avg recall: " + str(avgRecall))
    print("Avg f1: " + str(avgF1))
    print()

def filterTextCorpus(corpusDict, language):

    filterdCorpusDict = {}

    for id, sent in corpusDict.items():
        filterdSentence = sent.lower()
        filterdSentence = re.sub('[^A-Za-z0-9 ]+', '', filterdSentence)
        filterdSentence = removeStopWords(filterdSentence, language)
        filterdSentence = stemmSentence(filterdSentence)

        if filterdSentence is "":
            continue

        filterdCorpusDict[id] = filterdSentence

    return filterdCorpusDict

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

def summarizeDocument(corpus, edgeWeightMethod, priorMethod , language="portuguese"):

    corpusDict = {}
    for id, sent in enumerate(corpus):
        corpusDict[id] = sent

    filteredCorpus = filterTextCorpus(corpusDict, language)

    if priorMethod == PriorMethod.UNIFORM:
        prior = generateUniformPrior(filteredCorpus)
    if priorMethod == PriorMethod.POSITION:
        prior = generatePositionPrior(filteredCorpus)
    if priorMethod == PriorMethod.TFIDF:
        prior = generateTfIdfPrior(filteredCorpus)
    if priorMethod == PriorMethod.BAYES:
        prior = generateBayesPrior(filteredCorpus)

    tfidfMatrix = simpleTfIdf(filteredCorpus)

    if edgeWeightMethod == WeightMethod.UNIFORM:
        simGraph = buildSimilarityGraph(tfidfMatrix, 0.10, False)
    if edgeWeightMethod == WeightMethod.TFIDF:
        simGraph = buildSimilarityGraph(tfidfMatrix, 0.10, True)


    pageRankRes = pageRank(simGraph, prior, 0.15, 50)

    sortedList = sorted(pageRankRes.items(), key=lambda x: x[1], reverse=True)
    top5Results = sortedList[:5]
    top5Results = sorted(top5Results, key=lambda x: x[0])

    summarizedResult = []
    for result in top5Results:
        summarizedResult.append(corpus[result[0]])

    return summarizedResult


def evaluateTextSummarization(allDocuments, allIdealDocuments, methodName, weightMethod, uniformMethod):
    evaluationSummary = []

    for idx, document in enumerate(allDocuments):
        corpus = sent_tokenize(document)
        summarizedDocument = summarizeDocument(corpus, weightMethod, uniformMethod)
        precision, recall, f1, ap = evaluateResults(allIdealDocuments[idx], summarizedDocument)
        evaluationSummary.append((precision, recall, f1, ap))

    printEvaluationSummary(evaluationSummary, methodName)


allDocuments = readAllTextFiles("TeMario/TeMario-ULTIMA VERSAO out2004/Textos-fonte/Textos-fonte com título/")
allIdealDocuments = readAllTextFiles("TeMario/TeMario-ULTIMA VERSAO out2004/Sumários/Extratos ideais automáticos/")

evaluateTextSummarization(allDocuments, allIdealDocuments, "uniform weighted - uniform prior page rank", WeightMethod.UNIFORM, PriorMethod.UNIFORM)
evaluateTextSummarization(allDocuments, allIdealDocuments, "tfidf weighted - uniform prior page rank", WeightMethod.TFIDF, PriorMethod.UNIFORM)
evaluateTextSummarization(allDocuments, allIdealDocuments, "tfidf weighted - pos prior page rank", WeightMethod.TFIDF, PriorMethod.POSITION)
evaluateTextSummarization(allDocuments, allIdealDocuments, "tfidf weighted - tfidf prior page rank", WeightMethod.TFIDF, PriorMethod.TFIDF)
evaluateTextSummarization(allDocuments, allIdealDocuments, "tfidf weighted - bayes prior page rank", WeightMethod.TFIDF, PriorMethod.BAYES)