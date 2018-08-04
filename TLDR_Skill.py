import re
import string
import numpy as np
import requests
import json
import nltk
import gensim
from collections import Counter
from flask import Flask
from flask_ask import Ask, statement, question
import requests

app = Flask(__name__)
ask = Ask(app, "/")

punc_regex = re.compile('[{}]'.format(re.escape(string.punctuation)))

def strip_punc(corpus):
    return punc_regex.sub('', corpus)

def to_counter(doc):
    """ 
    Produce word-count of document, removing all punctuation
    and removing all punctuation.
    
    Parameters
    ----------
    doc : str
    
    Returns
    -------
    collections.Counter
        lower-cased word -> count"""
    return Counter(strip_punc(doc).lower().split())

def to_bag(counters):
    """ 
    [word_counter0, word_counter1, ...] -> sorted list of unique words
    
    Parameters
    ----------
    counters : Iterable[collections.Counter]
        An iterable containing {word -> count} counters for respective
        documents.
    
    Returns
    -------
    List[str]
        An alphabetically-sorted list of all of the unique words in `counters`"""
    bag = set()
    for counter in counters:
        bag.update(counter)
    return sorted(bag)

def to_tf(counter, bag):
    """
    Generate tf descriptors for a counter given word bag

    Parameters
    ----------
        counter : collections.Counter
            Counter of words in a document

        bag : List
            Bag of words in correct order

    Returns
    -------
        tf : np.ndarray
            ndarray representing tf descriptor for the document.
    """
    return np.array([counter[word] for word in bag], dtype=float)

def to_idf(bag, counters):
    """ 
    Given the bag-of-words, and the word-counts for each document, computes
    the inverse document-frequency (IDF) for each term in the bag.
    
    Parameters
    ----------
    bag : Sequence[str]
        Ordered list of words that we care about

    counters : Iterable[collections.Counter]
        The word -> count mapping for each document.
    
    Returns
    -------
    numpy.ndarray
        An array whose entries correspond to those in `bag`, storing
        the IDF for each term `t`: 
                           log10(N / nt)
        Where `N` is the number of documents, and `nt` is the number of 
        documents in which the term `t` occurs.
    """
    N = len(counters)
    nt = [sum(1 if t in counter else 0 for counter in counters) for t in bag]
    nt = np.array(nt, dtype=float)
    return np.log10(N / nt)

def tldr(corpus, threshold=10):
    """ Overall function to consolidate a document via tf-idf sorting.
    
        Parameters
        ----------
            corpus : String
                Full text to summarize.
                
            threshold : int, optional(default=10)
                Number of sentences to include in the summary.
                If this value exceeds the number of sentences in
                the corpus, the entire text will be returned
                unchanged.
                
        Returns
        -------
            summary : String
                Summarized form of initial input text.
    """
    
    #Tokenize the corpus into sentences
    sentences = nltk.tokenize.sent_tokenize(corpus)
    
    if(threshold >= len(sentences)):
        return corpus
    
    #Remove punctuation and lower strings for tf-idf counting
    sentences_lower = [strip_punc(token).lower() for token in sentences]

    #Generate counters for each sentence and bag of words
    counters = []
    for sent in sentences_lower:
        counters.append(to_counter(sent))
        
    bag = to_bag(counters)
    
    #Compute tf descriptors for each sentence and idf descriptor
    tfs = np.ndarray((len(sentences), len(bag)))
    for i, counter in enumerate(counters):
        tfs[i,:] = to_tf(counter, bag)
        
    idf = to_idf(bag, counters)
    
    #Compute tf-idf scores, summing along each sentence, and re-sort sentence array
    tf_idf = tfs * idf
    sentence_scores = np.mean(tf_idf, axis=1)
    sentence_ids = sorted(np.argsort(sentence_scores)[::-1][:threshold])
    
    #Guarantee inclusion of first sentence in the document
    if 0 not in sentence_ids:
        sentence_ids.insert(0, 0)
        sentence_ids.pop()
    for i in sentence_ids:
        print(sentences[i] + "\n")
    #Return joined form of all top sentences
    return '\n'.join(sentences[i].split(".")[0] for i in sentence_ids)

@ask.launch
def start_skill():
    """
    Initial starting point for the TLDR Alexa skill.
    """
    msg = "Hello. What would you like information about?"
    return question(msg)

@ask.intent("SearchIntent")
def wiki_search(Search):
    """
    Perform a wikipedia API query and search for a given article.

    Parameters
    ----------
        Search : string
            Input query from the search intent.
    """
    
    #Capitalize search string and generate wikipedia API query
    page = Search.title()
    query = "https://en.wikipedia.org/w/api.php?action=query&format=json&titles="
    query += page
    query += "&prop=extracts&explaintext"

    #Query wikipedia API to obtain JSON
    response = requests.get(query).json()

    #Parse json to obtain extracted text
    page_data = next(iter(response['query']['pages'].values()))
    page_extract = ""

    #Attempt to get extracted data and return on failure
    try:
        page_extract = page_data['extract']
    except KeyError as e:
        return statement("Sorry, I haven't heard of that.")

    #Apply TLDR algorithm to extracted data
    page_tldr = tldr(page_extract, threshold=2) 

    return statement(page_tldr)

if __name__ == '__main__':
    app.run(debug=True)