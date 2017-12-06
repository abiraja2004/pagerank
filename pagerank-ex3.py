from collections import Counter
import math
import numpy
import glob
import re
import nltk
import random
import matplotlib.pyplot as plt
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from numpy import linalg
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def readAllTextFiles(pathToFiles):
    files = glob.glob(pathToFiles + "/**/*.txt", recursive=True)
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

#-----------------------------------------------------------------------
#TFIDF methods
#-----------------------------------------------------------------------
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
        normalizedWordCount = generateDocumentVector(sentence)
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



#---------------------------------------------------------------------
#PageRank methods:
#---------------------------------------------------------------------

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

        pageRankDict = pageRankCurIteration

        if iteration == 49:
            return pageRankDict

    return pageRankDict

#---------------------------------------------------------------------
#---------------------------------------------------------------------

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


def generatePageRankScore(corpusDict):
    prior = generatePositionPrior(corpusDict)

    tfidfMatrix = simpleTfIdf(corpusDict)
    simGraph = buildSimilarityGraph(tfidfMatrix, 0.14, False)
    pageRankScore = pageRank(simGraph, prior, 0.15, 1000)
    return pageRankScore


def generateTrainingFeatureDocuments(textList, idealTextList):

    allTrainingFeautres = []

    for idx in range(len(textList)):
        features = generateTrainingFeatureVector(textList[idx], idealTextList[idx])
        allTrainingFeautres += features

    return allTrainingFeautres

def generateFeatureDocuments(textList):
    allFeautres = []

    for idx in range(len(textList)):
        features = generateFeatureVector(textList[idx])
        allFeautres.append(features)

    return allFeautres


def generateFeatureVector(corpus):
    documentFeatureVector = []
    corpusDict = {}

    for id, sent in enumerate(corpus):
        corpusDict[id] = sent

    filteredCorpus = filterTextCorpus(corpusDict, "portuguese")

    positionFeatures = generatePositionPrior(filteredCorpus)
    tfidfFeatures = generateTfIdfPrior(filteredCorpus)
    bayesFeatures = generateBayesPrior(filteredCorpus)
    pageRankFeatures = generatePageRankScore(filteredCorpus)


    for id in positionFeatures.keys():
        documentFeatureVector.append((id, [positionFeatures[id], tfidfFeatures[id], bayesFeatures[id], pageRankFeatures[id]]))

    return documentFeatureVector


def generateTrainingFeatureVector(text, idealText):
    documentFeatureVector = []

    corpusDict = {}
    corpus = nltk.sent_tokenize(text)
    for id, sent in enumerate(corpus):
        corpusDict[id] = sent

    filteredCorpus = filterTextCorpus(corpusDict, "portuguese")

    idealCorpus = sent_tokenize(idealText)

    targetValueVec = []
    for idx, sentence in enumerate(corpus):
        if sentence in idealCorpus:
            targetValueVec.append(1.0)
        else:
            targetValueVec.append(0.0)

    positionFeatures = generatePositionPrior(filteredCorpus)
    tfidfFeatures = generateTfIdfPrior(filteredCorpus)
    bayesFeatures = generateBayesPrior(filteredCorpus)
    pageRankFeatures = generatePageRankScore(filteredCorpus)

    for id in positionFeatures.keys():
        #documentFeatureVector.append(([positionFeatures[id], tfidfFeatures[id], bayesFeatures[id], pageRankFeatures[id]], targetValueVec[id]))
        documentFeatureVector.append(([positionFeatures[id], tfidfFeatures[id], pageRankFeatures[id]], targetValueVec[id]))
        #documentFeatureVector.append(([tfidfFeatures[id], pageRankFeatures[id], positionFeatures[id]], targetValueVec[id]))

    return documentFeatureVector

def predictModel(features, weights):
    output = 0.0
    for (feature, weight) in zip(features, weights):
        output = output + feature * weight

    #threshold
    if output >= 0:
        output = 1.0
    else:
        output = 0.0

    return output

def predictModelVal(features, weights):
        output = 0

        # add bias to feature vector
        features.insert(0, 1.0)

        for feature, weight in zip(features, weights):
            output = output + feature * weight

        return output

def adjustWeights(weights, features, output, target):
    teachingStep = 0.01
    adjustedWeights = []

    for (weight, feature) in zip(weights, features):
        newWeight = weight + teachingStep*(target - output)*feature
        adjustedWeights.append(newWeight)

    return adjustedWeights

def predictFeaturesPerceptron(testFeaturesList, weights):
    allPredictionList = []

    for testFeatures in testFeaturesList:
        docPredictionList = []
        for features in testFeatures:
            predictionValue = predictModelVal(features[1], weights)
            docPredictionList.append((features[0], predictionValue))

        allPredictionList.append(docPredictionList)

    return allPredictionList

def trainPerceptronModel(trainingFeatures):

    mse = 999

    #add bias to feature vectors
    for featureRow in trainingFeatures:
        featureRow[0].insert(0, 1.0)

    featuresCount = len(trainingFeatures[0][0])

    #init weights:
    weights = []
    for idx in range(featuresCount):
        weights.append(random.random())

    epochs = 0
    #train model:
    while (math.fabs(mse-0.001)) > 0.0001:
        mse = 0.0
        error = 0.0

        for featureRow in trainingFeatures:
            output = predictModel(featureRow[0], weights)
            error = error + math.fabs(output - featureRow[1])
            weights = adjustWeights(weights, featureRow[0], output, featureRow[1])

        mse = error / len(trainingFeatures)
        epochs += 1
        #print("The mean square error of "+  str(epochs) + " epoch is "+ str(mse));

        if epochs == 1000:
            return weights

    return weights

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


def evaluateTextSummarizationPerceptron(allDocuments, allIdealDocuments, weights):
    evaluationSummary = []

    for idx, document in enumerate(allDocuments):
        corpus = sent_tokenize(document)
        summarizedDocument = summarizeDocumentPerceptron(corpus, weights)
        precision, recall, f1, ap = evaluateResults(allIdealDocuments[idx], summarizedDocument)
        evaluationSummary.append((precision, recall, f1, ap))

    printEvaluationSummary(evaluationSummary, "perceptron")

def summarizeDocumentPerceptron(corpus, weights):

    featuresSent = generateFeatureVector(corpus)
    docPrecitionRanking = []

    for features in featuresSent:
        predictionValue = predictModelVal(features[1], weights)
        docPrecitionRanking.append((features[0], predictionValue))

    sortedList = sorted(docPrecitionRanking, key=lambda x: x[1], reverse=True)
    top5Results = sortedList[:5]
    top5Results = sorted(top5Results, key=lambda x: x[0])

    summarizedResult = []
    for result in top5Results:
        summarizedResult.append(corpus[result[0]])

    return summarizedResult

allDocumentsTraining = readAllTextFiles("TeMário 2006/Originais/")
allIdealDocumentsTraining = readAllTextFiles("TeMário 2006/SumáriosExtractivos/")

featureVecTraining = generateTrainingFeatureDocuments(allDocumentsTraining, allIdealDocumentsTraining)
weights = trainPerceptronModel(featureVecTraining)

allDocumentsTest = readAllTextFiles("TeMario/TeMario-ULTIMA VERSAO out2004/Textos-fonte/Textos-fonte com título/")
allIdealDocumentsTest = readAllTextFiles("TeMario/TeMario-ULTIMA VERSAO out2004/Sumários/Extratos ideais automáticos/")

evaluateTextSummarizationPerceptron(allDocumentsTest, allIdealDocumentsTest, weights)
