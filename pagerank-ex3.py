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

#-----------------------------------------------------------------------
#TFIDF methods----------------------------------------------------------
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



#---------------------------------------------------------------------
#-----------------------------------------------------------------------

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


def generateTrainingFeatureVector(text, idealText):

    features = generateFeatureVector(text)



def generateFeatureVector(text):

    documentFeaterVector = []

    positionFeatures = generatePositionPrior(text)
    tfidfFeatures = generateTfIdfPrior(text)
    bayesFeatures = generateBayesPrior(text)




allDocuments = readAllTextFiles("TeMario/TeMario-ULTIMA VERSAO out2004/Textos-fonte/Textos-fonte com título/")
allIdealDocuments = readAllTextFiles("TeMario/TeMario-ULTIMA VERSAO out2004/Sumários/Extratos ideais automáticos/")

featureVec = generateTrainingFeatureVector(allDocuments[0], allIdealDocuments[0])