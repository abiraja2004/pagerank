from collections import Counter
import math
import numpy
import glob
import re
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from numpy import linalg

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
        #normalizedWordCount = generateDocumentVectorImproved(sentence, True)
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


def generateDocumentVectorImproved(document, normalize = False):
    tokenizer = RegexpTokenizer(r'\w+')

    stop_words = set(stopwords.words("portuguese"))
    tokens = tokenizer.tokenize(document)

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

    wordCount = Counter(filteredAndStemmedTokens)

    if normalize == True:
        wordCount = normalizeTermFrequency(wordCount)

    return wordCount

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
    documentN = len(documentCorpus)

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

def generatePositionPrior(text):
    prior = {}
    corpus = sent_tokenize(text)

    N = len(corpus)
    for pos in range(len(corpus)):
        # smoothing goes here:
        prior[pos] = N / (pos + 1)

    return prior


def generateBayesPrior(text):
    prior = {}

    documentCorpus = sent_tokenize(text)
    docCount = len(documentCorpus)
    documentTfMatrix = []

    for document in documentCorpus:
        documentTf = generateDocumentVectorImproved(document)
        documentTfMatrix.append(documentTf)

    queryTf = generateDocumentVectorImproved(text)

    for idx, documentTf in enumerate(documentTfMatrix):
        probQuery = 0
        docLength = sum(documentTf.values())

        for queryTerm, queryTF in queryTf.items():
            docTF = documentTf.get(queryTerm, 0)
            probDoc = math.pow(((docTF + 1) / (docLength + 1)), queryTF)
            probQuery = probQuery + math.log(1 / docCount) + math.log(probDoc)

        prior[idx] = probQuery

    return prior


def generateTfIdfPrior(text):
    prior = {}

    vocabularySet = set()
    termInDocuments = {}
    documentCorpus = sent_tokenize(text)

    # count of sentences (used in idf calculation)
    documentN = len(documentCorpus)

    documentsWordCounts = generateDocumentWordCountAndVocabulary(documentCorpus, vocabularySet)
    vocabulary = list(vocabularySet)

    generateTermCountsPerDocument(documentsWordCounts, vocabulary, termInDocuments)
    tfMatrix = generateTfMatrixPerDocument(documentsWordCounts, vocabulary)
    # generate tfidf array for each sentence:
    tfidfMatrix = generateTfIdfMatrixPerDocument(tfMatrix, vocabulary, termInDocuments, documentN)

    queryTfIdfArray = generateTfIdfArray(text, vocabulary, termInDocuments, documentN)
    prior = compareSimilarity(queryTfIdfArray, tfidfMatrix)

    return prior


def generatePageRankScore(text):
    prior = generateBayesPrior(text)
    corpus = sent_tokenize(text)
    tfidfMatrix = simpleTfIdf(corpus)
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
    allTrainingFeautres = []

    for idx in range(len(textList)):
        features = generateFeatureVector(textList[idx])
        allTrainingFeautres += features

    return allTrainingFeautres

def generateFeatureVector(text):
    documentFeatureVector = []

    positionFeatures = generatePositionPrior(text)
    tfidfFeatures = generateTfIdfPrior(text)
    bayesFeatures = generateBayesPrior(text)
    pageRankFeatures = generatePageRankScore(text)

    for idx in range(len(positionFeatures)):
        documentFeatureVector.append([positionFeatures[idx], tfidfFeatures[idx], bayesFeatures[idx], pageRankFeatures[idx]])

    return documentFeatureVector


def generateTrainingFeatureVector(text, idealText):
    documentFeatureVector = []

    corpus = sent_tokenize(text)
    idealCorpus = sent_tokenize(idealText)

    targetValueVec = []
    for idx, sentence in enumerate(corpus):
        if sentence in idealCorpus:
            targetValueVec.append(1.0)
        else:
            targetValueVec.append(0.0)

    positionFeatures = generatePositionPrior(text)
    tfidfFeatures = generateTfIdfPrior(text)
    bayesFeatures = generateBayesPrior(text)
    pageRankFeatures = generatePageRankScore(text)

    for idx in range(len(positionFeatures)):
        documentFeatureVector.append(([positionFeatures[idx], tfidfFeatures[idx], bayesFeatures[idx], pageRankFeatures[idx]], targetValueVec[idx]))

    return documentFeatureVector


allDocumentsTraining = readAllTextFiles("TeMário 2006/Originais/")
allIdealDocumentsTraining = readAllTextFiles("TeMário 2006/SumáriosExtractivos/")

allDocumentsTest = readAllTextFiles("TeMario/TeMario-ULTIMA VERSAO out2004/Textos-fonte/Textos-fonte com título/")
allIdealDocumentsTest = readAllTextFiles("TeMario/TeMario-ULTIMA VERSAO out2004/Sumários/Extratos ideais automáticos/")

featureVecTraining = generateTrainingFeatureDocuments(allDocumentsTraining, allIdealDocumentsTraining)
featureVecTest = generateFeatureDocuments(allDocumentsTest)

print("test")