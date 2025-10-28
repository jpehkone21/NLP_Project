"""
This script implements the task 4 from the project description
"""
from gensim import corpora
import matplotlib.pyplot as plt
from collections import defaultdict



"""
4. Now we want to track the yearly occurrent of the 5 most frequent terms and the 5 most highest TF-IDF
scores. For this purpose, suggest a script that draws on the same plot the yearly evolution of the
frequency of each of these five terms in each dataset.
"""

top5_most_frequent = []
top5_tfidf = []

# extracting the top 5 words:
with open("project/top_50_frequent_1.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]
    top5_most_frequent.append([line.split()[0] for line in lines[:5]])

with open("project/top_50_frequent_2.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]
    top5_most_frequent.append([line.split()[0] for line in lines[:5]])

with open("project/top_50_frequent_3.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]
    top5_most_frequent.append([line.split()[0] for line in lines[:5]])

with open("project/top_50_frequent_4.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]
    top5_most_frequent.append([line.split()[0] for line in lines[:5]])



with open("project/top_50_tfidf_1.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]
    top5_tfidf.append([line.split()[0] for line in lines[:5]])

with open("project/top_50_tfidf_2.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]
    top5_tfidf.append([line.split()[0] for line in lines[:5]])

with open("project/top_50_tfidf_3.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]
    top5_tfidf.append([line.split()[0] for line in lines[:5]])

with open("project/top_50_tfidf_4.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]
    top5_tfidf.append([line.split()[0] for line in lines[:5]])


print(top5_most_frequent)
print(top5_tfidf)

# downloading dictionaries and corpuses for each dataset:
dictionary1 = corpora.Dictionary.load("project/dict_dataset_1")
print("dictionary1 loaded")
corpus1 = corpora.MmCorpus("project/corpus_dataset_1.mm")
print("corpus1 loaded")

dictionary2 = corpora.Dictionary.load("project/dict_dataset_2")
print("dictionary2 loaded")
corpus2 = corpora.MmCorpus("project/corpus_dataset_2.mm")
print("corpus2 loaded")

dictionary3 = corpora.Dictionary.load("project/dict_dataset_3")
print("dictionary3 loaded")
corpus3 = corpora.MmCorpus("project/corpus_dataset_3.mm")
print("corpus3 loaded")

dictionary4 = corpora.Dictionary.load("project/dict_dataset_4")
print("dictionary4 loaded")
corpus4 = corpora.MmCorpus("project/corpus_dataset_4.mm")
print("corpus4 loaded")

#processing each dataset separately:

total_frequencies_dataset_1 = {}
for i in range(len(top5_most_frequent)):

    #target_words = top5_most_frequent[i]
    target_words = top5_tfidf[i]

    total_counts = defaultdict(int)
    for doc in corpus1:
        for token_id, count in doc:
            total_counts[token_id] += count
    for word in target_words:
        token_id = dictionary1.token2id.get(word)
        if token_id is not None:
            total_frequencies_dataset_1[word] = total_counts[token_id]
            print(f"{word}: {total_counts[token_id]}")
        else:
            print(f"{word}: not found in dictionary")
print("dataset 1 processed: ")
print(total_frequencies_dataset_1)


total_frequencies_dataset_2 = {}
for i in range(len(top5_most_frequent)):

    #target_words = top5_most_frequent[i]
    target_words = top5_tfidf[i]

    total_counts = defaultdict(int)
    for doc in corpus2:
        for token_id, count in doc:
            total_counts[token_id] += count
    for word in target_words:
        token_id = dictionary2.token2id.get(word)
        if token_id is not None:
            total_frequencies_dataset_2[word] = total_counts[token_id]
            print(f"{word}: {total_counts[token_id]}")
        else:
            print(f"{word}: not found in dictionary")
print("dataset 2 processed: ")
print(total_frequencies_dataset_2)




total_frequencies_dataset_3 = {}
for i in range(len(top5_most_frequent)):

    #target_words = top5_most_frequent[i]
    target_words = top5_tfidf[i]

    total_counts = defaultdict(int)
    for doc in corpus3:
        for token_id, count in doc:
            total_counts[token_id] += count

    for word in target_words:
        token_id = dictionary3.token2id.get(word)
        if token_id is not None:
            total_frequencies_dataset_3[word] = total_counts[token_id]
            print(f"{word}: {total_counts[token_id]}")
        else:
            print(f"{word}: not found in dictionary")
print("dataset 3 processed: ")
print(total_frequencies_dataset_3)


total_frequencies_dataset_4 = {}
for i in range(len(top5_most_frequent)):

    #target_words = top5_most_frequent[i]
    target_words = top5_tfidf[i]

    total_counts = defaultdict(int)
    for doc in corpus4:
        for token_id, count in doc:
            total_counts[token_id] += count
    for word in target_words:
        token_id = dictionary4.token2id.get(word)
        if token_id is not None:
            total_frequencies_dataset_4[word] = total_counts[token_id]
            print(f"{word}: {total_counts[token_id]}")
        else:
            print(f"{word}: not found in dictionary")
print("dataset 4 processed: ")
print(total_frequencies_dataset_4)



# constructing the plot:
counts_per_year = [total_frequencies_dataset_1, total_frequencies_dataset_2, total_frequencies_dataset_3, total_frequencies_dataset_4]
years = ["2011-2018", "2019-2020", "2021", "2022-2024"]
words = list(total_frequencies_dataset_1.keys())

word_trends = {word: [] for word in words}

for year_counts in counts_per_year:
    for word in words:
        word_trends[word].append(year_counts.get(word, 0))

plt.figure(figsize=(10, 6))

for word, counts in word_trends.items():
    plt.plot(years, counts, marker='o', label=word)

plt.title("Word frequencies over the years")
plt.xlabel("Years")
plt.ylabel("Count")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()