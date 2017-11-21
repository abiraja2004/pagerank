import nltk
from collections import Counter
import math
import numpy
import glob
import re
from numpy import linalg
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize


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
    tfidfMatrix = []
    for tfArray in tfMatrix:
        tfdidfArray = calcTfIdfArray(tfArray, vocabulary, termInDocuments, documentN)
        tfidfMatrix.append(tfdidfArray)

    return tfidfMatrix

def getRetrievedSentences(text, topSentencesIdx):
    retrievedDocuments = []
    corpus = sent_tokenize(text)

    for docIdx in topSentencesIdx:
        retrievedDocuments.append(corpus[docIdx])

    return retrievedDocuments


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

def compareMMRSimilarity(docTfIdfArray, tfidfMatrix):
    mmrResults = []
    mmrIdxList = []
    lambdaMmr = 0.3

    for count in enumerate(tfidfMatrix):
        simResults = []
        for idx, tfidfArray in enumerate(tfidfMatrix):

            if idx in mmrIdxList:
                continue

            sim = cosineSimilarity(docTfIdfArray, tfidfArray)
            mmr = ((1- lambdaMmr) * sim) - (lambdaMmr * sumSimularityMMR(tfidfArray, tfidfMatrix, mmrIdxList))

            simResults.append((idx, mmr))

        # sorted by hightes similarity socre
        simResults = sorted(simResults, key=lambda x: x[1], reverse=True)
        mmrResults.append(simResults[0])
        mmrIdxList.append(simResults[0][0])

    return mmrResults

def sumSimularityMMR(tfidfArray, tfidfMatrix, idxList):
    sim = 0
    for idx in idxList:
        selectedTfIdfArray = tfidfMatrix[idx]
        sim += cosineSimilarity(tfidfArray, selectedTfIdfArray)

    return sim

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

def simpleTfIdf1SingleDoc(documentCorpus, query):
    vocabularySet = set()
    termInDocuments = {}

    # count of sentences (used in idf calculation)
    documentN = len(documentCorpus)

    documentsWordCounts = generateDocumentWordCountAndVocabulary(documentCorpus,  vocabularySet)
    vocabulary = list(vocabularySet)

    generateTermCountsPerDocument(documentsWordCounts, vocabulary, termInDocuments)

    # generate tf array for each sentence:
    tfMatrix = generateTfMatrixPerDocument(documentsWordCounts, vocabulary)

    # generate tfidf array for each sentence:
    tfidfMatrix = generateTfIdfMatrixPerDocument(tfMatrix, vocabulary, termInDocuments, documentN)

    # generate document tfidf Array from query corresponding to the same vector space model:
    tokens = nltk.word_tokenize(query)
    wordCount = Counter(tokens)
    normalizedWordCount = normalizeTermFrequency(wordCount)
    queryTfArray = calcTfArray(normalizedWordCount, vocabulary)
    queryTfIdfArray = calcTfIdfArray(queryTfArray, vocabulary, termInDocuments, documentN)

    # perform similarity operations of query with all documents in the corpus:
    simResults = compareMMRSimilarity(queryTfIdfArray, tfidfMatrix)

    return simResults

def simpleTfIdf1(allDocuments, allIdealDocuments):
    evaluationSummary = []

    for idx, document in enumerate(allDocuments):
        corpus = sent_tokenize(document)
        simResults = simpleTfIdf1SingleDoc(corpus, document)

        sortedSimReults = sorted(simResults, key=lambda x: x[1], reverse=True)
        topSentencesIdx = []

        topSentencesIdx.append(sortedSimReults[0][0])
        topSentencesIdx.append(sortedSimReults[1][0])
        topSentencesIdx.append(sortedSimReults[2][0])
        topSentencesIdx.append(sortedSimReults[3][0])
        topSentencesIdx.append(sortedSimReults[4][0])

        # top 3 but shown as they appear in original document
        topSentencesIdx = sorted(topSentencesIdx)

        # sorted by occurence in document
        topSentencesIdx = sorted(topSentencesIdx)
        retrievedDocuments = getRetrievedSentences(document, topSentencesIdx)

        precision, recall, f1, ap = evaluateResults(allIdealDocuments[idx], retrievedDocuments)
        evaluationSummary.append((precision, recall, f1, ap))

    printEvaluationSummary(evaluationSummary, "TfIdf MMR")


def firstNSentences(allDocuments, allIdealDocuments, N):

    evaluationSummary = []

    for index, text in enumerate(allDocuments):
        # best 5 sentences
        topSentencesIdx = []
        for i in range(N):
            topSentencesIdx.append(i)

        retrievedDocuments = getRetrievedSentences(text, topSentencesIdx)
        precision, recall, f1, ap = evaluateResults(allIdealDocuments[index], retrievedDocuments)
        evaluationSummary.append((precision, recall, f1, ap))

    printEvaluationSummary(evaluationSummary, "First 5 Sentences")





allDocuments = readAllTextFiles("./TeMario/TeMario-ULTIMA VERSAO out2004/Textos-fonte/Textos-fonte com título/")
allIdealDocuments = readAllTextFiles("./TeMario/TeMario-ULTIMA VERSAO out2004/Sumários/Extratos ideais automáticos/")

simpleTfIdf1(allDocuments, allIdealDocuments)
firstNSentences(allDocuments, allIdealDocuments, 5)


