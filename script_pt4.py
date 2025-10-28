"""
This script implements tasks 6, 7, 8, 9 and 10 from the project description
"""

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import matplotlib.pyplot as plt
from numpy import dot, mean
from numpy.linalg import norm
from gensim.models import Word2Vec
import numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.manifold import TSNE


def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            yield simple_preprocess(line) 


"""
6. Use the Gensim library to train separate Word2Vec models for each dataset.
"""

# read files for dataset 1:
for year in range(2011, 2019):
    print("reading year ", year)
    for i in range(1, 13):
        filename = f"{i:02}.txt"
        data_path = "project/extracted_texts/" + str(year) + "/" + filename
        sentences = list(read_file(data_path))

model_1 = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4,
    sg=1 
)

model_1.save("project/word2vec_dataset1.model")
print("model 1 saved")

# read files for dataset 2:
for year in range(2019, 2021):
    print("reading year ", year)
    for i in range(1, 13):
        filename = f"{i:02}.txt"
        data_path = "project/extracted_texts/" + str(year) + "/" + filename
        sentences = list(read_file(data_path))

model_2 = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4,
    sg=1 
)

model_2.save("project/word2vec_dataset2.model")
print("model 2 saved")

# read files for dataset 3:
for year in range(2021, 2022):
    print("reading year ", year)
    for i in range(1, 13):
        filename = f"{i:02}.txt"
        data_path = "project/extracted_texts/" + str(year) + "/" + filename
        sentences = list(read_file(data_path))

model_3 = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4,
    sg=1 
)

model_3.save("project/word2vec_dataset3.model")
print("model 3 saved")

# read files for dataset 4:
for year in range(2022, 2025):
    print("reading year ", year)
    for i in range(1, 13):
        filename = f"{i:02}.txt"
        data_path = "project/extracted_texts/" + str(year) + "/" + filename
        sentences = list(read_file(data_path))

model_4 = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4,
    sg=1 
)

model_4.save("project/word2vec_dataset4.model")
print("model 4 saved")

#print(model.wv.most_similar('politiikka', topn=10))
#print(model.wv['urheilu'])


#load models if they exist
""" 
model_1 = Word2Vec.load("project/word2vec_dataset1.model")
model_2 = Word2Vec.load("project/word2vec_dataset2.model")
model_3 = Word2Vec.load("project/word2vec_dataset3.model")
model_4 = Word2Vec.load("project/word2vec_dataset4.model")
"""

def align_models(base_model, other_model):

    common_vocab = list(set(base_model.wv.index_to_key) & set(other_model.wv.index_to_key))
    
    base_vecs = np.array([base_model.wv[word] for word in common_vocab])
    other_vecs = np.array([other_model.wv[word] for word in common_vocab])
    
    R, _ = orthogonal_procrustes(other_vecs, base_vecs)
    
    other_model.wv.vectors = np.dot(other_model.wv.vectors, R)
    return other_model


# align all other models with the first one
base = model_1 
model_2 = align_models(base, model_2)
model_3 = align_models(base, model_3)
model_4 = align_models(base, model_4)

all_models = {
    "dataset_1": model_1, 
    "dataset_2": model_2, 
    "dataset_3": model_3, 
    "dataset_4": model_4, 
    }


def cosine_similarity(vec1, vec2):
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))


def normalized_similarity(word, model1, model2, stable_words):
    base_sim = cosine_similarity(model1.wv[word], model2.wv[word])
    stable_sims = [cosine_similarity(model1.wv[w], model2.wv[w]) for w in stable_words if w in model1.wv and w in model2.wv]
    #print(np.mean(stable_sims))
    return base_sim / mean(stable_sims)

def get_neighbors(model, word, topn=30):
    if word in model.wv:
        return [w for w, _ in model.wv.most_similar(word, topn=topn)]
    else:
        return []
    
def neighborhood_overlap(neighbors1, neighbors2):
    if not neighbors1 or not neighbors2:
        return None
    overlap = len(set(neighbors1) & set(neighbors2))
    union = len(set(neighbors1) | set(neighbors2))
    return overlap / union  # Jaccard similarity




"""
7. Choose at least five target words (e.g., tekoäly, ilmasto, politiikka) and calculate the cosine similarity of
their embedding vectors between different time periods. Normalize the similarity scores using a set of
stable function words such as “minä”, “ja”.
"""

target_words = ['tekoäly', 'ilmasto', 'politiikka', 'urheilu', 'suomi']



"""
10. Repeat task 5 (assuming this was meant to be task 7. ) for the list of frequent words obtained in task 2.
"""
# some of the most frequent words from datasets
# target_words = ['myös', 'jo', 'voi', 'sanoo', 'kertoo']

stable_words = ['minä', 'ja'] 



for word in target_words:
    if word in model_1.wv and word in model_2.wv:
        sim1 = cosine_similarity(model_1.wv[word], model_2.wv[word])
        norm_sim = normalized_similarity(word, model_1, model_2, stable_words)
        print(f"{word}: similarity between datasets 1 and 2 = {sim1:.3f}, normalized = {norm_sim:.3f}")

    if word in model_2.wv and word in model_3.wv:
        sim1 = cosine_similarity(model_2.wv[word], model_3.wv[word])
        norm_sim = normalized_similarity(word, model_2, model_3, stable_words)
        print(f"{word}: similarity between datasets 2 and 3 = {sim1:.3f}, normalized = {norm_sim:.3f}")
        
    
    if word in model_3.wv and word in model_4.wv:
        sim1 = cosine_similarity(model_3.wv[word], model_4.wv[word])
        norm_sim = normalized_similarity(word, model_3, model_4, stable_words)
        print(f"{word}: similarity between datasets 3 and 4 = {sim1:.3f}, normalized = {norm_sim:.3f}")
        

    if word in model_1.wv and word in model_4.wv:
        sim1 = cosine_similarity(model_1.wv[word], model_4.wv[word])
        norm_sim = normalized_similarity(word, model_1, model_4, stable_words)
        print(f"{word}: similarity between datasets 1 and 4 = {sim1:.3f}, normalized = {norm_sim:.3f}")
        

    print("\n")


"""
8. Perform a neighborhood change analysis for the target words by comparing the nearest neighbors across
models.
"""
for word in target_words:
    n1 = get_neighbors(model_1, word)
    n2 = get_neighbors(model_2, word)
    n3 = get_neighbors(model_3, word)
    n4 = get_neighbors(model_4, word)
    score1 = neighborhood_overlap(n1, n2)
    print(f"{word}: neighborhood similarity between datasets 1 and 2 = {score1:.2f}")
    score2 = neighborhood_overlap(n2, n3)
    print(f"{word}: neighborhood similarity between datasets 2 and 3 = {score2:.2f}")
    score3 = neighborhood_overlap(n3, n4)
    print(f"{word}: neighborhood similarity between datasets 3 and 4 = {score3:.2f}")
    score4 = neighborhood_overlap(n1, n4)
    print(f"{word}: neighborhood similarity between datasets 1 and 4 = {score4:.2f}")

    print("\n")


"""
9. Use t-SNE or a similar method to display the neighborhoods of the target words in a 2D space
"""
for word in target_words:

    embeddings = []
    labels = []
    colors = []
    color_map = ['red', 'blue', 'green', 'orange']

    for i, (dataset, model) in enumerate(all_models.items()):
        neighbors = get_neighbors(model, word)
        words = [word] + neighbors
    
        for w in words:
            embeddings.append(model.wv[w])
            labels.append(w)
            colors.append(color_map[i])


    embeddings = np.array(embeddings)
    tsne = TSNE(n_components=2, random_state=42, perplexity=15, init='pca')
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(labels):
        plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], color=colors[i], s=50)
        plt.text(embeddings_2d[i, 0]+0.01, embeddings_2d[i, 1]+0.01, label, fontsize=8)

    # add legend manually
    for i, period in enumerate(all_models.keys()):
        plt.scatter([], [], color=color_map[i], label=period)
    plt.legend()
    plt.title(f"Neighborhood of '{word}'")
    plt.axis('off')
    plt.show()
            
