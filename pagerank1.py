def readGraph():
    file = open("pri_links.txt", "r")
    lines = file.read().splitlines()

    graph = {}

    for line in lines:
        splitedLine = line.split(' ')

        for i,docId in enumerate(splitedLine):
            if docId.isalnum() == False:
                continue

            if i == 0:
                key = docId
                graph[key] = []
                continue

            list = graph.get(key, [])
            list.append(docId)
            graph[key] = list

    return graph


def invertLinkFromGraph(linkFromGraph):
    invertedGraph = {}

    for key, values in linkFromGraph.items():
        for val in values:
            list = invertedGraph.get(val,[])
            list.append(key)
            invertedGraph[val] = list

    return invertedGraph

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

def pageRank(linkGraphDict, dampingFactor, numberOfIterations):
    pageRankDict = {}
    allIterations = []
    allDiffIterations = []
    invertedGraph = invertGraph(linkGraphDict)

    N = len(linkGraphDict.keys())

    #init pageRankvector
    for node in linkGraphDict.keys():
        pageRankDict[node] = 1.0/ N

    for iteration in range(numberOfIterations):
        pageRankCurIteration = {}

        for node, values in linkGraphDict.items():

            linkSum = 0
            for linkFromNode in values:
                linkSum = linkSum +  pageRankDict[linkFromNode] / len(invertedGraph[linkFromNode])

            rank = (dampingFactor / N) + (1 - dampingFactor) * linkSum
            pageRankCurIteration[node] = rank

        diffDict = compareDifferenceDict(pageRankDict, pageRankCurIteration)
        pageRankDict = pageRankCurIteration
        allIterations.append(pageRankCurIteration)
        allDiffIterations.append(diffDict)

        isConverged = checkConvergence(diffDict, 0.000000000001)
        if isConverged:
            print("Number of iterations to converge: " + str(iteration))
            return pageRankDict



    return pageRankDict



graph = readGraph()
invertedGraph = invertGraph(graph)
pageRankRes = pageRank(invertedGraph, 0.2, 50)
print(pageRankRes)