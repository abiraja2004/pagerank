import nltk
from collections import Counter
import math
import numpy
from numpy import linalg
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize
import re
import glob
from nltk.corpus import stopwords

def readTextFile(filename):
    file = open(filename, "r")
    textData = file.read()
    return textData


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

def calcBm25TfArray(termCountDoc, avgdl, vocabulary):
    tfArray = []
    k1 = 1.2
    b = 0.75
    docLength = sum(termCountDoc.values(), 0.0)
    for vocab in vocabulary:
        tf = termCountDoc.get(vocab, 0)
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * (docLength/avgdl))
        bm25Tf = numerator / denominator
        tfArray.append(bm25Tf)

    return tfArray


def calcBm25TfIdfArray(tfArray, vocabulary, termInDocuments, documentN):
    tfdidfArray = []

    for index, tf in enumerate(tfArray):
        vocab = vocabulary[index]
        termCountGlobal = termInDocuments[vocab]
        idfValue = math.log(((documentN - termCountGlobal +0.5) / ( 1 + termCountGlobal)))
        tfidfValue = tf * idfValue
        tfdidfArray.append(tfidfValue)

    return tfdidfArray


def normalizeTermFrequency(wordCounts):
    maxFreq = max(wordCounts.values())
    for key in wordCounts:
        wordCounts[key] /= maxFreq

    return wordCounts

def generateDocumentVector(document, normalize = True):
    # remove punctuation, lemmatation, stemming
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(document)

    wordCount = Counter(tokens)
    if normalize == True:
        wordCount = normalizeTermFrequency(wordCount)

    return wordCount

def generateDocumentVectorImproved(document, normalize = True):
    tokenizer = RegexpTokenizer(r'\w+')

    stop_words = set(stopwords.words("portuguese"))
    tokens = tokenizer.tokenize(document)

    #tagged = posTagger.tag(tokens)
    filterd_tokens = []
    for token in tokens:
        if token not in stop_words:
            token = token.lower()
            filterd_tokens.append(token)

    filteredAndStemmedTokens = []
    stemmer = nltk.stem.RSLPStemmer()
    for token in filterd_tokens:
        stemmed = stemmer.stem(token)
        filteredAndStemmedTokens.append(stemmed)

    bigrm = list(nltk.bigrams(filteredAndStemmedTokens))

    wordCount = Counter(filteredAndStemmedTokens)
    wordCount += Counter(bigrm)

    if normalize == True:
        wordCount = normalizeTermFrequency(wordCount)

    return wordCount

def generateDocumentWordCountAndVocabulary(documentCorpus,  vocabularySet):
    documentWordCounts = []
    # generate vocabulary and term count per document
    for sentence in documentCorpus:
        normalizedWordCount = generateDocumentVector(sentence)
        vocabularySet.update(normalizedWordCount.keys())
        documentWordCounts.append(normalizedWordCount)

    return documentWordCounts

def generateDocumentWordCountAndVocabularyBm25(documentCorpus,  vocabularySet):
    documentWordCounts = []
    # generate vocabulary and term count per document
    for sentence in documentCorpus:
        normalizedWordCount = generateDocumentVectorImproved(sentence, False)
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

def generateBm25TfMatrixPerDocument(documentsWordCounts, avgdl, vocabulary):
    tfMatrix = []
    for counterDoc in documentsWordCounts:
        tfArray = calcBm25TfArray(counterDoc, avgdl, vocabulary)
        tfMatrix.append(tfArray)

    return tfMatrix

def generateTfIdfMatrixPerDocument(tfMatrix, vocabulary, termInDocuments, documentN):
    tfidfMatrix = []
    for tfArray in tfMatrix:
        tfdidfArray = calcTfIdfArray(tfArray, vocabulary, termInDocuments, documentN)
        tfidfMatrix.append(tfdidfArray)

    return tfidfMatrix

def generateBm25TfIdfMatrixPerDocument(tfMatrix, vocabulary, termInDocuments, documentN):
    tfidfMatrix = []
    for tfArray in tfMatrix:
        tfdidfArray = calcBm25TfIdfArray(tfArray, vocabulary, termInDocuments, documentN)
        tfidfMatrix.append(tfdidfArray)

    return tfidfMatrix

def getRetrievedSentences(text, topSentencesIdx):
    retrievedDocuments = []
    corpus = sent_tokenize(text)

    for docIdx in topSentencesIdx:
        retrievedDocuments.append(corpus[docIdx])

    return retrievedDocuments

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

def averageDocLength(documentCountList):
    avgdl = 0
    totalSentencesCount = 0
    for sentences in documentCountList:
        length = sum(sentences.values(), 0.0)
        avgdl += length

    totalSentencesCount += len(documentCountList)

    avgdl = avgdl / totalSentencesCount
    return avgdl

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
    simResults = []
    for index, tfidfArray in enumerate(tfidfMatrix):
        sim = cosineSimilarity(queryTfIdfArray, tfidfArray)
        simResults.append((index, sim))

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

    printEvaluationSummary(evaluationSummary, "TfIdf1")


def simpleTfIdfBm25(allDocuments, allIdealDocuments):
    evaluationSummary = []

    for idx, document in enumerate(allDocuments):
        corpus = sent_tokenize(document)
        simResults = simpleTfIdfBm25SingleDoc(corpus, document)

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

    printEvaluationSummary(evaluationSummary, "Bm25")

def simpleTfIdfBm25SingleDoc(documentCorpus, query):
    vocabularySet = set()
    termInDocuments = {}

    # count of sentences (used in idf calculation)
    documentN = len(documentCorpus)

    documentsWordCounts = generateDocumentWordCountAndVocabularyBm25(documentCorpus,  vocabularySet)
    vocabulary = list(vocabularySet)

    avgdl = averageDocLength(documentsWordCounts)

    generateTermCountsPerDocument(documentsWordCounts, vocabulary, termInDocuments)

    # generate tf array for each sentence:
    tfMatrix = generateBm25TfMatrixPerDocument(documentsWordCounts, avgdl, vocabulary)

    # generate tfidf array for each sentence:
    tfidfMatrix = generateBm25TfIdfMatrixPerDocument(tfMatrix, vocabulary, termInDocuments, documentN)

    # generate document tfidf Array from query corresponding to the same vector space model:
    normalizedWordCount = generateDocumentVectorImproved(query, False)

    queryTfArray = calcBm25TfArray(normalizedWordCount, avgdl, vocabulary)
    queryTfIdfArray = calcBm25TfIdfArray(queryTfArray, vocabulary, termInDocuments, documentN)

    # perform similarity operations of query with all documents in the corpus:
    simResults = []
    for index, tfidfArray in enumerate(tfidfMatrix):
        sim = cosineSimilarity(queryTfIdfArray, tfidfArray)
        simResults.append((index, sim))

    return simResults






allDocuments = readAllTextFiles("./TeMario/TeMario-ULTIMA VERSAO out2004/Textos-fonte/Textos-fonte com título/")
allIdealDocuments = readAllTextFiles("./TeMario/TeMario-ULTIMA VERSAO out2004/Sumários/Extratos ideais automáticos/")

simpleTfIdf1(allDocuments, allIdealDocuments)
simpleTfIdfBm25(allDocuments, allIdealDocuments)
