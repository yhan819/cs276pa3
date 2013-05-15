import sys
import re
import os, glob, os.path
from math import log

#inparams
#  featureFile: input file containing queries and url features
#return value
#  queries: map containing list of results for each query
#  features: map containing features for each (query, url, <feature>) pair
def extractFeatures(featureFile):
    f = open(featureFile, 'r')
    queries = {}
    features = {}

    for line in f:
      key = line.split(':', 1)[0].strip()
      value = line.split(':', 1)[-1].strip()
      if(key == 'query'):
        query = value
        queries[query] = []
        features[query] = {}
      elif(key == 'url'):
        url = value
        queries[query].append(url)
        features[query][url] = {}
      elif(key == 'title'):
        features[query][url][key] = value
      elif(key == 'header'):
        curHeader = features[query][url].setdefault(key, [])
        curHeader.append(value)
        features[query][url][key] = curHeader
      elif(key == 'body_hits'):
        if key not in features[query][url]:
          features[query][url][key] = {}
        temp = value.split(' ', 1)
        features[query][url][key][temp[0].strip()] \
                    = [int(i) for i in temp[1].strip().split()]
      elif(key == 'body_length' or key == 'pagerank'):
        features[query][url][key] = int(value)
      elif(key == 'anchor_text'):
        anchor_text = value
        if 'anchors' not in features[query][url]:
          features[query][url]['anchors'] = {}
      elif(key == 'stanford_anchor_count'):
        features[query][url]['anchors'][anchor_text] = int(value)
      
    f.close()
    return (queries, features) 

#inparams
#  queries: map containing list of results for each query
#  features: map containing features for each query,url pair
#return value
#  rankedQueries: map containing ranked results for each query
def baseline(queries, features, dfDict, totalDocNum):
    rankedQueries = {}
    for query in queries.keys():
      results = queries[query]
      #features[query][x].setdefault('body_hits', {}).values() returns the list of body_hits for all query terms
      #present in the document, empty if nothing is there. We sum over the length of the body_hits array for all
      #query terms and sort results in decreasing order of this number
      rankedQueries[query] = sorted(results, 
                                    key = lambda x: sum([len(i) for i in 
                                    features[query][x].setdefault('body_hits', {}).values()]), reverse = True)

    return rankedQueries


#inparams
#  queries: contains ranked list of results for each query
#  outputFile: output file name
def printRankedResults(queries):
    for query in queries:
      print("query: " + query)
      for res in queries[query]:
        print("  url: " + res)

def getIdf():
  term_id_f = "word.dict"
  posting_f = "posting.dict"
  doc_f = "doc.dict"

  allqueryFile = "AllQueryTerms"
  queryTermsDict = {}
  docNum = 0
  word_dict = {}
  doc_freq_dict = {}
  
  file = open(allqueryFile, 'r')
  for line in file.readlines():
    queryTermsDict[line.strip()] = 0

  file = open(doc_f, 'r')
  for l in file.readlines():
    docNum += 1
  print >> sys.stderr, "totalNum = " + str(docNum)
  
  file = open(term_id_f, 'r')
  for line in file.readlines():
    parts = line.split('\t')
    word_dict[int(parts[1])] = parts[0]
  file = open(posting_f, 'r') 
  for line in file.readlines():
    parts = line.split('\t')
    term_id = int(parts[0])
    doc_freq = int(parts[2])
    doc_freq_dict[word_dict[term_id]] = doc_freq
  
  return (docNum, doc_freq_dict)

#inparams
#  featureFile: file containing query and url features
def main(featureFile):
    #output file name
    outputFile = "ranked.txt" #Please don't change this!

    #populate map with features from file
    (queries, features) = extractFeatures(featureFile)

    #get idf values
    (totalDocNum, dfDict) = getIdf()

    #calling baseline ranking system, replace with yours
    rankedQueries = baseline(queries, features, dfDict, totalDocNum)
    
    #print ranked results to file
    printRankedResults(rankedQueries)
       
if __name__=='__main__':
    if (len(sys.argv) < 2):
      print "Insufficient number of arguments" 
    main(sys.argv[1])
