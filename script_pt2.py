"""
This script implements tasks 2 and 3 from the project description.
"""
from itertools import product
from wordcloud import WordCloud
import matplotlib.pyplot as plt

top50_most_frequent = []
top50_tfidf = []

with open("project/top_50_frequent_1.txt", "r", encoding="utf-8") as f:
    top50_most_frequent.append([line.split()[0] for line in f if line.strip()])

with open("project/top_50_frequent_2.txt", "r", encoding="utf-8") as f:
    top50_most_frequent.append([line.split()[0] for line in f if line.strip()])

with open("project/top_50_frequent_3.txt", "r", encoding="utf-8") as f:
    top50_most_frequent.append([line.split()[0] for line in f if line.strip()])

with open("project/top_50_frequent_4.txt", "r", encoding="utf-8") as f:
    top50_most_frequent.append([line.split()[0] for line in f if line.strip()])



with open("project/top_50_tfidf_1.txt", "r", encoding="utf-8") as f:
    top50_tfidf.append([line.split()[0] for line in f if line.strip()])

with open("project/top_50_tfidf_2.txt", "r", encoding="utf-8") as f:
    top50_tfidf.append([line.split()[0] for line in f if line.strip()])

with open("project/top_50_tfidf_3.txt", "r", encoding="utf-8") as f:
    top50_tfidf.append([line.split()[0] for line in f if line.strip()])

with open("project/top_50_tfidf_4.txt", "r", encoding="utf-8") as f:
    top50_tfidf.append([line.split()[0] for line in f if line.strip()])


"""
2. Calculate the Jaccard similarity score to quantify the similarity 
between the datasets from the 50 mostfrequent word perspectives and 50 most 
highest TF-IDF scores. Present the result in 4 x 4 matrix. 
"""


# for most frequent words:
matrix = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]] #4x4 matrix

for i, j in product(range(4), range(4)):
    set_i = top50_most_frequent[i]
    set_j = top50_most_frequent[j]
    intersection = len(set(set_i).intersection(set(set_j)))
    union = len(set(set_i).union(set(set_j)))
    matrix[i][j] = round((intersection / union), 4)

print("Jaccard's similarity matrix (most frequent words): ")
for i in range(4):
    print(matrix[i])

# for top tfidf scores:
matrix = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]] #4x4 matrix

for i, j in product(range(4), range(4)):
    set_i = top50_tfidf[i]
    set_j = top50_tfidf[j]
    intersection = len(set(set_i).intersection(set(set_j)))
    union = len(set(set_i).union(set(set_j)))
    matrix[i][j] = round((intersection / union), 4)


print("Jaccard's similarity matrix(top tfidf scores): ")
for i in range(4):
    print(matrix[i])


"""
3. Present an illustration of the content of the most frequent words and highest TF-IDF scores using
wordCloud illustrations for each dataset
"""

# Wordclouds for most frequent words:
for i in range(4):
    string = (" ").join(top50_most_frequent[i])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(string)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  
    plt.title(f"Top 50 most frequent words in dataset {i+1}")
    plt.show()

# Wordclouds for highest tfidf scores:
for i in range(4):
    string = (" ").join(top50_tfidf[i])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(string)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  
    plt.title(f"Top 50 words with highest tfidf scores in dataset {i+1}")
    plt.show()
