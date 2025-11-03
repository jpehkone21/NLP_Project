"""
This script implements the task 1 of the project description.
"""
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import corpora, models
from collections import Counter
import heapq
import html

stop_words = set(stopwords.words('finnish'))
#print(stop_words)

datasets = []

def preprocess(text):
    tokens = word_tokenize(text.lower())  # lowercase and tokenize
    tokens = [t for t in tokens if t.isalpha()]  # remove punctuation/numbers
    tokens = [t for t in tokens if t not in stop_words]  # remove stopwords
    return tokens


def save_dict_and_corpus(start, end, dataset_nro):
    # Goes through all monthly files from 2011-2024 (total 168 files)

    # Goes through all monthly files from 2011-2018
    for i in range(start, end): #total: 14
        year = 2010 + i
        #print(year)
        for i in range(1, 13):
            filename = f"{i:02}.txt"
            data_path = "project/extracted_texts/" + str(year) + "/" + filename
            print(data_path)
            with open(data_path, "r", encoding="utf-8") as f:
                text = f.read()
            text = html.unescape(text) #fix some fo the ä and ö letters (e.g. ensimm&auml;inen)
            tokens = preprocess(text)
            datasets.append(tokens)

            
    # generate dictionary and tf-idf model and save them
    dictionary = corpora.Dictionary(datasets)
    corpora.Dictionary.save(dictionary, f"project/dict_dataset_{dataset_nro}")

    corpus = [dictionary.doc2bow(text) for text in datasets]
    corpora.MmCorpus.serialize(f"project/corpus_dataset_{str(dataset_nro)}.mm", corpus)
    
    return dictionary, corpus


# Run the code separately for each dataset:

dictionary, corpus = save_dict_and_corpus(1, 9, 1) #start 2011, end 2018, dataset 1
#dictionary, corpus = save_dict_and_corpus(9, 11, 2) #start 2019, end 2020, dataset 2
#dictionary, corpus = save_dict_and_corpus(11, 12, 3) #start 2021, end 2021, dataset 3
#dictionary, corpus = save_dict_and_corpus(12, 15, 4) #start 2022, end 2024, dataset 4



#dictionary = corpora.Dictionary.load("project/dict_dataset_1")
#corpus = corpora.MmCorpus("project/corpus_dataset_1.mm")



"""
1. Suggest a script that extracts the tokens and vocabulary of each dataset after stopword removal using the
default NLTK list for Finnish language. Then use the Gensim TF-IDF vectorizer to generate the
dictionary and TF-IDF score of each token, considering all datasets) as well as the 50 most frequent
tokens (excluding Stopwords) and 50 tokens with highest TF-IDF scores. 
"""


tfidf_model = models.TfidfModel(corpus)
corpus_tfidf = tfidf_model[corpus]


# 50 most frequent tokens:
all_tokens = [token for text in datasets for token in text]
freq_dist = Counter(all_tokens)
most_common_tokens = freq_dist.most_common(50)

print("\nTop 50 most frequent tokens:")
for token, count in most_common_tokens:
    print(f"{token}: {count}")



# 50 tokens with highest tf-idf score
all_tfidf = []
for doc in corpus_tfidf:
    for id, score in doc:
        all_tfidf.append((dictionary[id], score))

top_tfidf_tokens = heapq.nlargest(50, all_tfidf, key=lambda x: x[1])

print("\nTop 50 tokens with highest TF-IDF scores:")
for token, score in top_tfidf_tokens:
    print(f"{token}: {score:.4f}")



#Saving the results in txt files
with open("project/top_50_frequent_1.txt", "w", encoding="utf-8") as f:
    for token, count in most_common_tokens:
        f.write(f"{token}\t{count}\n")

with open("project/top_50_tfidf_1.txt", "w", encoding="utf-8") as f:
    for token, score in top_tfidf_tokens:
        f.write(f"{token}\t{score:.6f}\n")




