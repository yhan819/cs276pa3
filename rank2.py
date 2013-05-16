import sys
import re
import math

#inparams
#  featureFile: input file containing queries and url features
#return value
#  queries: map containing list of results for each query
#  features: map containing features for each (query, url, <feature>) pair
def extractFeatures(featureFile):
    f = open(featureFile, 'r')
    queries = {}
    features = {}
    count = {}
    count["num_url"] = 0
    count["len_url"] = 0
    count["len_title"] = 0
    count["len_header"] = 0
    count["len_body"] = 0
    count["len_anchor"] = 0

    for line in f:
      key = line.split(':', 1)[0].strip()
      value = line.split(':', 1)[-1].strip()
      if(key == 'query'):
        query = value
        queries[query] = []
        features[query] = {}
      elif(key == 'url'):
        url = value
        count["num_url"] += 1
        count["len_url"] += len(value)
        queries[query].append(url)
        features[query][url] = {}
      elif(key == 'title'):
        features[query][url][key] = value
        count["len_title"] += len(value.split())
      elif(key == 'header'):
        curHeader = features[query][url].setdefault(key, [])
        curHeader.append(value)
        features[query][url][key] = curHeader
        count["len_header"] += len(value.split())
      elif(key == 'body_hits'):
        if key not in features[query][url]:
          features[query][url][key] = {}
        temp = value.split(' ', 1)
        features[query][url][key][temp[0].strip()] \
                    = [int(i) for i in temp[1].strip().split()]
      elif(key == 'body_length' or key == 'pagerank'):
        features[query][url][key] = int(value)
        if key == 'body_length':
          count["len_body"] += int(value)
      elif(key == 'anchor_text'):
        anchor_text = value
        count["len_anchor"] += len(anchor_text.split())
        if 'anchors' not in features[query][url]:
          features[query][url]['anchors'] = {}
      elif(key == 'stanford_anchor_count'):
        features[query][url]['anchors'][anchor_text] = int(value)
      
    f.close()
    return (queries, features, count) 

#inparams
#  queries: map containing list of results for each query
#  features: map containing features for each query,url pair
#return value
#  rankedQueries: map containing ranked results for each query
def baseline(queries, features, dfDict, totalDocNum, count):
    rankedQueries = {}


    avgurl = count["len_url"] * 1.0 / count["num_url"] 
    avgtitle = count["len_title"] * 1.0 / count["num_url"]
    avgheader = count["len_header"] * 1.0 / count["num_url"]
    avgbody = count["len_body"] * 1.0 / count["num_url"]
    avganchor = count["len_anchor"] * 1.0 / count["num_url"]

    # Parameters for tf counts for doc
    b_url = 1
    b_title = 1
    b_header = 1
    b_body = 1
    b_anchor = 1
    w_url = 1
    w_title = 1
    w_header = 1
    w_body = 1
    w_anchor = 1
    lamb = 2
    lamb_p = 1
    k_1 = 5

    for query in queries.keys():
      results = queries[query]
      terms = query.split()
      scores = {}
      for doc_url in results:
        url = re.findall(r"['\w']+:", doc_url)
        doc_score = 0.0
        for t in terms:
          wdt = 0.0
          info = features[query][doc_url]
          #ftf for url
          fturl = 0.0
          tfurl = 0
          for u in url:
            if t == u:
              tfurl += 1
          fturl += 1.0 * tfurl / (1 + b_url * ((len(doc_url) / avgurl) - 1))
          wdt += w_url * fturl 
          #ftf for title
          fttitle = 0.0
          tftitle = 0
          t_l = info["title"].split()
          for word in t_l:
            if word == t:
              tftitle += 0
          if len(t_l) > 0:
            fttitle += 1.0 * tftitle / (1 + b_title * ((len(t_l) / avgtitle) - 1))
            wdt += w_title * fttitle
          #tft for header
          if "header" in info:
            ftheader = 0.0
            tfheader = 0
            lenhead = 0
            for header in info["header"]:
              lenhead += len(header.split())
              for word in header.split():
                if word == t:
                  tfheader += 1
            ftheader += 1.0 * tfheader / (1 + b_header * ((lenhead / avgheader) - 1))
            wdt += w_header * ftheader
          # ftf for body
          if "body_hits" in info:
            ftbody = 0.0
            tfbody = 0
            if t in info["body_hits"]:
              tfbody = len(info["body_hits"][t])
            ftbody += 1.0 * tfbody / (1 + b_body * ((info["body_length"] / avgbody) - 1))
            wdt += w_body * ftbody
          # ftf for anchor
          if "anchor" in info:
            ftanchor = 0.0
            tfanchor = 0
            anchor_len = 0
            for text in info["anchors"]:
              c_p_a = 0
              anchor_len += len(text.split())
              for word in text.split():
                if word == t:
                  c_p_a += 1
              tfanchor += c_p_a * info["anchors"][text]
            ftanchor += 1.0 * tfanchor / (1 + b_anchor * ((anchor_len / avganchor) - 1))
            wdt += w_anchor * ftanchor
          
          #nontextual: pagerank
          nont = lamb * 1.0 * math.log(lamb_p + info["pagerank"])
 
          #idf
          if t not in dfDict:
            df = 1
          else:
            df = dfDict[t] + 1
          idf = math.log((totalDocNum + 1)/df)
          
          doc_score += wdt * idf / (k_1 + wdt) + nont
        scores[doc_url] = doc_score
      rankedQueries[query] = sorted(results, key = lambda x: scores[x], reverse=True) 
    return rankedQueries

# getIdf gets returns a total number of doc and doc_freq_dict
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
#  queries: contains ranked list of results for each query
#  outputFile: output file name
def printRankedResults(queries):
    for query in queries:
      print("query: " + query)
      for res in queries[query]:
        print("  url: " + res)

#inparams
#  featureFile: file containing query and url features
def main(featureFile):
    #output file name
    outputFile = "ranked.txt" #Please don't change this!

    #populate map with features from file
    (queries, features, count) = extractFeatures(featureFile)

    #get idf values
    (totalDocNum, dfDict) = getIdf()

    #calling baseline ranking system, replace with yours
    rankedQueries = baseline(queries, features, dfDict, totalDocNum, count)
    
    #print ranked results to file
    printRankedResults(rankedQueries)
       
if __name__=='__main__':
    if (len(sys.argv) < 2):
      print "Insufficient number of arguments" 
    main(sys.argv[1])
