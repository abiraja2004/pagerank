from urllib.request import urlopen
import xml.etree.ElementTree as ET
from collections import Counter
import math
import numpy
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
        allIterations.append(pageRankCurIteration)

        if iteration == 49:
            return pageRankDict

    return pageRankDict


def summarizeDocument(corpus, edgeWeightMethod, priorMethod , language="portuguese"):

    corpusDict = {}
    for id, sent in enumerate(corpus):
        corpusDict[id] = sent[1]

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
        simGraph = buildSimilarityGraph(tfidfMatrix, 0.14, False)
    if edgeWeightMethod == WeightMethod.TFIDF:
        simGraph = buildSimilarityGraph(tfidfMatrix, 0.14, True)

    pageRankRes = pageRank(simGraph, prior, 0.15, 1000)

    sortedList = sorted(pageRankRes.items(), key=lambda x: x[1], reverse=True)

    every4thResults = sortedList[0::10]
    top5Results = every4thResults[:5]
    top5Results = sorted(top5Results, key=lambda x: x[0])

    summarizedResult = []
    for result in top5Results:
        summarizedResult.append(corpus[result[0]])

    return summarizedResult


def fetchRSSText(url, label):
    site = urlopen(url)
    content = site.read()
    site.close()

    root = ET.fromstring(content)
    items = root.findall(".//item")

    sentenceList = []

    for item in items:
        titleNode = item.find("./title")
        descNode = item.find("./description")
        title = titleNode.text
        desc = descNode.text
        sentenceList.append((label, title))
        desc = re.sub('<[^<]+?>', '', desc)
        descSentences = sent_tokenize(desc)
        for sent in descSentences:
            sentenceList.append((label, sent))

    sentenceList = [x for x in sentenceList if x[1] is not None]
    return sentenceList


def fetchNewsCorpus():
    nytimesUrl = "http://rss.nytimes.com/services/xml/rss/nyt/World.xml"
    nytimesNews = fetchRSSText(nytimesUrl, "NyTimes")

    washingtonpostUrl = "http://feeds.washingtonpost.com/rss/rss_blogpost"
    washingtonpostNews = fetchRSSText(washingtonpostUrl, "Washington Post")

    latimesUrl = "http://www.latimes.com/world/rss2.0.xml"
    latimesNews = fetchRSSText(latimesUrl, "LA Times")

    cnnUrl = "http://rss.cnn.com/rss/edition_world.rss"
    cnnNews = fetchRSSText(cnnUrl, "CNN")

    newsCorpus = []
    newsCorpus += nytimesNews
    newsCorpus += washingtonpostNews
    newsCorpus += latimesNews
    newsCorpus += cnnNews

    return newsCorpus


def genereateNewsHtml(newsSummary):
    file = open("news.html", "w")
    file.write("<html>\n")

    file.write("<h1>News summary</h1>\n")

    for sentence in newsSummary:
        file.write("<h2>\n")
        file.write(sentence[1] + " - <b><i>" + sentence[0] + "</i></b> \n")
        file.write("</h2><p>\n\n")

    file.write("</html>\n")
    file.close()


newsCorpus = fetchNewsCorpus()
summary = summarizeDocument(newsCorpus, WeightMethod.TFIDF, PriorMethod.TFIDF, 'english')
genereateNewsHtml(summary)

print(summary)
