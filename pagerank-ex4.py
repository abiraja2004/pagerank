from urllib.request import urlopen
import xml.etree.ElementTree as ET
from nltk.tokenize import sent_tokenize
import re
from pagerankEx2 import summarizeDocument
from pagerankEx2 import WeightMethod
from pagerankEx2 import PriorMethod


def fetchRSSText(url):
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
        sentenceList.append(title)
        desc = re.sub('<[^<]+?>', '', desc)
        descSentences = sent_tokenize(desc)
        sentenceList += descSentences

    sentenceList = [x for x in sentenceList if x is not None]

    return sentenceList

def fetchNewsCorpus():
    nytimesUrl = "http://rss.nytimes.com/services/xml/rss/nyt/World.xml"
    nytimesNews = fetchRSSText(nytimesUrl)

    washingtonpostUrl = "http://feeds.washingtonpost.com/rss/rss_blogpost"
    washingtonpostNews = fetchRSSText(washingtonpostUrl)

    latimesUrl = "http://www.latimes.com/world/rss2.0.xml"
    latimesNews = fetchRSSText(latimesUrl)

    cnnUrl = "http://rss.cnn.com/rss/edition_world.rss"
    cnnNews = fetchRSSText(cnnUrl)

    textCorpus = []
    textCorpus += nytimesNews
    textCorpus += washingtonpostNews
    textCorpus += latimesNews
    textCorpus += cnnNews

    return textCorpus


def genereateNewsHtml(newsSummary):

    file = open("news.html", "w")
    file.write("<html>\n")

    file.write("<h1>News summary</h1>\n")

    for sentence in newsSummary:
        file.write("<h2>\n")
        file.write(sentence + "\n")
        file.write("</h2>\n")


    file.write("</html>\n")
    file.close()


newsCorpus = fetchNewsCorpus()
summary = summarizeDocument(newsCorpus, WeightMethod.TFIDF, PriorMethod.POSITION)
genereateNewsHtml(summary)

print(summary)
