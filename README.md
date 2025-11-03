# Project 9: Historical Finnish Word Embeddings from News

Johanna Pehkonen [solo project]


### Explanations for the files

- download_data.py
  
  This script extracts text from  downloaded json datasets
Ignores all metadata, images etc. about articles and
saves the texts in their own monthly files.

- script_pt1.py
  
  Implements the task 1 of the project description:

- script_pt2.py
  
  Implements tasks 2 and 3 from the project description.
  
- script_pt3.py
  
  Implements the task 4 from the project description
  
- script_pt4.py
  
  Implements tasks 6, 7, 8, 9 and 10 from the project description




### How to run the project
Have the following files downloaded from https://www.kielipankki.fi/aineistot/ylenews/
- project/ylenews-fi-2011-2018-src/
- project/ylenews-fi-2019-2020-src/
- project/ylenews-fi-2021-src/
- project/ylenews-fi-2022-2024-src/

Run the script download_data.py to extract the text from the data and save it in text files for easier processing.

Next, run the script named script_pt1.py to construct dictionary and corpus for each dataset. This script will also detect the 50 most frequent words and 50 words with highest tf-idf scores and save them to separate text files which are needed later. Uncomment one dataset at a time from the code and run the script separately for each dataset.

Next, run script_pt2.py, which will read the 50 most frequent and highest tf-idf words from files and calculates the Jaccard's similarity matrixes, and contructs the word cloud visualizations. 

Next, run script_pt3.py, which uses the saved dictionaries and corpuses to track the evolution of frequencies of the most frequent and highest tf-idf words between the datasets and builds graphs of them. 

Lastly, running script_pt4.py will read the original extracted text files and train a word2vec model for each dataset, and compare word embeddings and neighbourhoods of five target words. 



### Used Python libraries
- nltk                3.9.1
- gensim              4.3.3
- wordcloud           1.9.4
- matplotlib          3.8.3
- numpy               1.26.4
- scipy               1.12.0
- scikit-learn        1.7.2

Python 3.12.2 was used  for the project.
