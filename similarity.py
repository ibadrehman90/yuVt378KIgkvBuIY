
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sentence_transformers import SentenceTransformer
import csv
import pandas as pd
import pandas as pandasForSortingCSV
import numpy as np
import re
import scipy
from nltk.corpus import stopwords

df = pd.read_excel("potential-talents.xlsx")
glove_df = pd.read_excel("potential-talents-glove.xlsx")
job_titles = df['job_title']
glove_job_titles = glove_df['job_title']
keyword = 'seeking human resources'
model_name = 'bert-base-nli-mean-tokens'
glove_res = []


model = SentenceTransformer(model_name)
tfidf_vectors = TfidfVectorizer().fit_transform([keyword] + job_titles)
bert_vec = model.encode([keyword] + job_titles)

def loadGloveModel(gloveFile):
    print ("Loading Glove Model")
    with open(gloveFile, encoding="utf8" ) as f:
        content = f.readlines()
    model = {}
    for line in content:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print ("Done.",len(model)," words loaded!")
    return model

# glove_model = loadGloveModel('glove.6b/glove.6B.300d.txt')

def preprocess(raw_text):

    # keep only words
    letters_only_text = re.sub("[^a-zA-Z]", " ", raw_text)

    # convert to lower case and split 
    words = letters_only_text.lower().split()

    # remove stopwords
    stopword_set = set(stopwords.words("english"))
    cleaned_words = list(set([w for w in words if w not in stopword_set]))

    return cleaned_words

def cosine_distance_between_two_words(word1, word2):
    return (1- scipy.spatial.distance.cosine(model[word1], model[word2]))

def calculate_heat_matrix_for_two_sentences(s1,s2):
    s1 = preprocess(s1)
    s2 = preprocess(s2)
    result_list = [[cosine_distance_between_two_words(word1, word2) for word2 in s2] for word1 in s1]
    result_df = pd.DataFrame(result_list)
    result_df.columns = s2
    result_df.index = s1
    return result_df


def similarity():
    cosine_similarities = linear_kernel(tfidf_vectors[0:1], tfidf_vectors).flatten()
    tfidf_fitness = [item.item() for item in cosine_similarities[1:]]
    # max_fitness = max(document_scores)
    # max_fitness_index = document_scores.index(max_fitness)
    # print("The candidate with the highest fitness is ", job_titles[max_fitness_index], " with the fitness of ", max_fitness)
    bert_fitness = np.array(cosine_similarity(
        [bert_vec[0]],
        bert_vec[1:]
    ))

    # uncomment this for GloVe embedding

    # for i in glove_job_titles:
    #     vector_1 = np.mean([glove_model[word] for word in preprocess(keyword)],axis=0)
    #     vector_2 = np.mean([glove_model[word] for word in preprocess(i)],axis=0)
    #     cosine = scipy.spatial.distance.cosine(vector_1, vector_2)
    #     glove_res.append(cosine)

    abc = np.array(glove_res)

    # saving the csv with all three types of embeddings
    # with open('seeking-human-resources-final.csv', 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(['titles','Tfidf', 'BERT', 'GloVe'])
    #     for x, y, z, k in zip(job_titles,tfidf_fitness,bert_fitness[0],abc):
    #         writer.writerow([x,y,z,k])
    
    # sorting csv to show candidates with highest fitness
    pd.set_option('display.max_rows', None)
    csvData = pandasForSortingCSV.read_csv("aspiring-human-resource-final.csv")
    csvData.sort_values(["BERT"],axis=0,ascending=[False],inplace=True)
  
    # displaying sorted data frame
    print("\nAfter sorting:")
    print(csvData)


if __name__ == "__main__":
    similarity()

