from collections import Counter
import math
import numpy
import glob
import re
from numpy import linalg
from enum import Enum
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize


class WeightMethod(Enum):
    UNIFORM = 1
    TFIDF = 2


class PriorMethod(Enum):
    UNIFORM = 1
    POSITION = 2


def readAllTextFiles(pathToFiles):
    files = glob.glob(pathToFiles + "/*.txt")
    documentTextList = []
    # iterate over the list getting each file
    for file in files:
        corpus = []
        # open the file and then call .read() to get the text
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


def generateDocumentVector(document):
    # remove punctuation, lemmatation, stemming
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(document)

    wordCount = Counter(tokens)
    normalizedWordCount = normalizeTermFrequency(wordCount)

    return normalizedWordCount


def generateDocumentWordCountAndVocabulary(documentCorpus, vocabularySet):
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

    documentsWordCounts = generateDocumentWordCountAndVocabulary(documentCorpus, vocabularySet)
    vocabulary = list(vocabularySet)

    generateTermCountsPerDocument(documentsWordCounts, vocabulary, termInDocuments)

    tfMatrix = generateTfMatrixPerDocument(documentsWordCounts, vocabulary)

    # generate tfidf array for each sentence:
    tfidfMatrix = generateTfIdfMatrixPerDocument(tfMatrix, vocabulary, termInDocuments, documentN)
    return tfidfMatrix


def invertGraph(graph):
    invertedGraph = {}

    for key in graph.keys():
        invertedGraph[key] = []

    for key, values in graph.items():
        for val in values:
            invertedGraph[val].append(key)

    return invertedGraph


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
    #invertedGraph = invertGraph(linkGraphDict)
    invertedGraph = linkGraphDict;

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
                for edge2 in invertedGraph[edge[0]]:
                    edgeSum = edgeSum + edge2[1]

                linkSum = linkSum + ((pageRankDict[edge[0]] * edge[1]) / edgeSum)

            priorProbability = (prior[node]/sum(prior.values()))
            rank = dampingFactor * priorProbability + (1 - dampingFactor) * linkSum
            pageRankCurIteration[node] = rank

        diffDict = compareDifferenceDict(pageRankDict, pageRankCurIteration)
        pageRankDict = pageRankCurIteration
        allIterations.append(pageRankCurIteration)
        allDiffIterations.append(diffDict)

        isConverged = checkConvergence(diffDict, 0.000000001)
        if isConverged:
            print("Number of iterations to converge: " + str(iteration))
            return pageRankDict
        if iteration == 49:
            return pageRankDict

    return pageRankDict


def generatePositionPrior(tfidfMatrix):
    prior = {}
    N = len(tfidfMatrix)
    for pos in tfidfMatrix.keys():
        # smoothing goes here:
        prior[pos] = N / (pos + 1)

    return prior


def generatePriorUnifrom(tfidfMatrix):
    prior = {}
    N = len(tfidfMatrix)
    for node in tfidfMatrix.keys():
        # smoothing goes here:
        prior[node] =  1

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


def summarizeDocument(text, edgeWeightMethod, priorMethod):
    corpus = sent_tokenize(text)

    tfidfMatrix = simpleTfIdf(corpus)

    if priorMethod == PriorMethod.UNIFORM:
        prior = generatePriorUnifrom(tfidfMatrix)
    if priorMethod == PriorMethod.POSITION:
        prior = generatePositionPrior(tfidfMatrix)

    if edgeWeightMethod == WeightMethod.UNIFORM:
        simGraph = buildSimilarityGraph(tfidfMatrix, 0.2, False)
    if edgeWeightMethod == WeightMethod.TFIDF:
        simGraph = buildSimilarityGraph(tfidfMatrix, 0.2, True)


    pageRankRes = pageRank(simGraph, prior, 0.15, 1000)

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
        summarizedDocument = summarizeDocument(document, weightMethod, uniformMethod)
        precision, recall, f1, ap = evaluateResults(allIdealDocuments[idx], summarizedDocument)
        evaluationSummary.append((precision, recall, f1, ap))

    printEvaluationSummary(evaluationSummary, methodName)


allDocuments = readAllTextFiles("TeMario/TeMario-ULTIMA VERSAO out2004/Textos-fonte/Textos-fonte com título/")
allIdealDocuments = readAllTextFiles("./TeMario/TeMario-ULTIMA VERSAO out2004/Sumários/Extratos ideais automáticos/")


evaluateTextSummarization(allDocuments, allIdealDocuments, "uniform page rank", WeightMethod.UNIFORM, PriorMethod.UNIFORM)
evaluateTextSummarization(allDocuments, allIdealDocuments, "tfidf weighted - pos prior page rank", WeightMethod.TFIDF, PriorMethod.POSITION)