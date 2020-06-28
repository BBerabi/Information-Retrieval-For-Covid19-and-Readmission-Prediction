import nltk
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from gensim import corpora
import os
import pickle
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

# Get data
PATH_CSV = os.path.abspath("./data/papers19-20.csv")
PATH_INDEX = './data/similarities.index'
PATH_DICTIONARY = './data/dictionary19-20.dict'
PATH_TFIDF = './data/tfidf.pkl'
PATH_PICKLE = './data/bow_corpus19-20.pickle'

data = pd.read_csv(PATH_CSV)
print("Got csv")
papers = data['text'].values

# Tokenize
punctuation = ",.?!()-_\"\'\\\n\r\t;:+*<>@#§^$%&|/"
processed = [[w.lower() for w in word_tokenize(document)] for document in papers]
print("Tokenized")

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
print("Lemmatized")
# Filter
processed = [[w for w in doc if (w not in stopwords.words('english')) and (w not in punctuation)] for doc in processed]
print("Filtered")

# Compute frequency
frequency = defaultdict(int)
for document in processed:
    for token in document:
        frequency[token] += 1
print("Frequencies computed")
# Get only words with frequency >1
processed_corpus = [[w for w in document if frequency[w] > 1] for document in processed]

# Save it into dictionary
dictionary = corpora.Dictionary(processed_corpus)
dictionary.save(PATH_DICTIONARY)
print("Dictionary created and saved into ", PATH_DICTIONARY)
# Create BoW vectors
bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]
print("BoW vectors are created")

with open(PATH_PICKLE,"wb") as f :
    pickle.dump(bow_corpus,f)



print("DONE")
