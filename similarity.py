
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import csv
import pandas as pd
import pandas as pandasForSortingCSV

df = pd.read_excel("potential-talents.xlsx")
job_titles = df['job_title']
keyword = 'Aspiring human resources'

doc_vectors = TfidfVectorizer().fit_transform([keyword] + job_titles)

def similarity():
    cosine_similarities = linear_kernel(doc_vectors[0:1], doc_vectors).flatten()
    document_scores = [item.item() for item in cosine_similarities[1:]]
    max_fitness = max(document_scores)
    max_fitness_index = document_scores.index(max_fitness)
    print("The candidate with the highest fitness is ", job_titles[max_fitness_index], " with the fitness of ", max_fitness)

    # saving the csv
    # with open('Aspiring-human-resources.csv', 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(['titles','fitness'])
    #     for x, y in zip(job_titles, document_scores):
    #         writer.writerow([x,y])

    # sorting csv to show candidates with highest fitness
    pd.set_option('display.max_rows', None)
    csvData = pandasForSortingCSV.read_csv("Aspiring-human-resources.csv")
    csvData.sort_values(["fitness"],axis=0,ascending=[False],inplace=True)
  
    # displaying sorted data frame
    print("\nAfter sorting:")
    print(csvData)


if __name__ == "__main__":
    similarity()

