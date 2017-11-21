import nltk
from collections import Counter
import math
import numpy
from numpy import linalg
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize

def readTextFile(filename):
    file = open(filename, "r")
    textData = file.read()
    return textData

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


def simpleTfIdf(documentCorpus, query):
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



pathToTextFile = "test.txt"

text = readTextFile(pathToTextFile)
corpus = sent_tokenize(text)

simResults = simpleTfIdf(corpus, text)
print(simResults)

sortedSimReults = sorted(simResults, key=lambda x: x[1], reverse=True)
top3SentenceIdx = []

top3SentenceIdx.append(sortedSimReults[0][0])
top3SentenceIdx.append(sortedSimReults[1][0])
top3SentenceIdx.append(sortedSimReults[2][0])

#top 3 but shown as they appear in original document
top3SentenceIdx = sorted(top3SentenceIdx)

#print 3 most similar sentences in order they appear in the document
for docIdx in top3SentenceIdx:
    print(corpus[docIdx])
    print()

