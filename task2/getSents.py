import pandas as pd
from gensim.corpora import Dictionary
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from gensim import corpora, similarities
from gensim import models
from nltk.corpus import wordnet
import nltk
import os
import pickle


def getBowVectors(papers,dictionary):
    # Tokenize
    punctuation = ",.?!()-_\"\'\\\n\r\t;:+*<>@#§^$%&|/"
    processed = [[w.lower() for w in word_tokenize(document)] for document in papers]


    def get_wordnet_pos(word):
        """Map POS tag to first character lemmatize() accepts"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}

        return tag_dict.get(tag, wordnet.NOUN)


    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    processed = [[lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in doc] for doc in processed]
    # Filter
    processed = [[w for w in doc if (w not in stopwords.words('english')) and (w not in punctuation)] for doc in processed]
    # Compute frequency
    frequency = defaultdict(int)
    for document in processed:
        for token in document:
            frequency[token] += 1
    # Get only words with frequency >1
    processed = [[w for w in document if frequency[w] > 1] for document in processed]
    bow_vectors = [dictionary.doc2bow(text) for text in processed]
    return bow_vectors



#Number of relevant papers to fetch from corpus
n_papers = 10
n_sentences = 10
#Get data
PATH_CSV = os.path.abspath("./data/papers19-20.csv")
PATH_INDEX = './data/similarities.index'
PATH_DICTIONARY = './data/dictionary19-20.dict'
PATH_TFIDF = './data/tfidf.pkl'
PATH_SENTENCES = './data/query_results.csv'
PATH_PICKLE = './data/bow_corpus19-20.pickle'

with open(PATH_PICKLE,"rb") as file :
    bow_corpus = pickle.load(file)


data = pd.read_csv(PATH_CSV)
print("Got csv")
papers = data['text']


# Load dictionary
dictionary = Dictionary.load(PATH_DICTIONARY)
print("Loaded dictionary")
tfidf = models.TfidfModel(bow_corpus, id2word=dictionary, normalize=True, slope=0.25)
# Similarity Index
# Use especially similarity but not matrix similarity, since
index = similarities.Similarity(None, tfidf[bow_corpus], num_features=len(dictionary))

def process_query(query):
    query = dictionary.doc2bow(query.lower().split(" "))
    query = tfidf[query]
    return query


query = "risk factor corona virus 2019"
bow_query = process_query(query)
print("Query procesed ")
#Get similar docs and sort them according to their value in descending order
similars = index[bow_query]
similars = sorted(enumerate(similars), key=lambda item: -item[1])

#Get  relevant paper_ids
ids = []

for i, s in enumerate(similars):
    ids.append(s[0])
    if len(ids) > n_papers:
        break

#After you have the relevant paper ids
#Get relevant papers' bow vectors
relevant_papers = data.iloc[ids,:]
bow_papers = getBowVectors(relevant_papers['text'],dictionary)

df = pd.DataFrame([],columns=['paper_id','sentence'])

#look deeper in each paper
for i in range(len(bow_papers)):
    sentences = []
    #Extract most important keywords of the document
    keyword_ids = [ s[0] for s in sorted(tfidf[bow_papers[i]],key=lambda tup: -tup[-1])][:n_sentences]

    keywords = []
    for j in keyword_ids:
        keywords.append(dictionary[j])

    # Get paper and split it into sentences
    tmp_paper_id = relevant_papers.iloc[i,0]
    tmp_paper = relevant_papers.iloc[i,1]
    tmp_paper = tmp_paper.split('.')
    #Find the sentences containing those words
    for sent in tmp_paper:
        for k in keywords:
            if k in sent:
                sentences.append([tmp_paper_id,sent])
                break

    df = df.append(pd.DataFrame(sentences,columns=['paper_id','sentence']),ignore_index=True)
print(df)
df.to_csv(PATH_SENTENCES)
print("Relevant sentences are saved into ",PATH_SENTENCES)



