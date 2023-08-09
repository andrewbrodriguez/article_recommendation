import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.text import Tokenizer
from sklearn.cluster import KMeans

df = pd.read_csv("/Users/andrewrodriguez/Desktop/article_recommendation/Articles.csv", encoding = "ISO-8859-1")
with open('20k.txt') as f:
    words = f.readlines()
corpus = []
for word in words:
    corpus.append([word[:-1]])
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)

matrices = tokenizer.texts_to_matrix(df['Article'], mode = 'count')
ohe = []
for i in range(2692):
    ohe.append(matrices[i][0:])

# Perform kmean clustering
num_clusters = 100
clustering_model = KMeans(n_clusters=num_clusters)
clustering_model.fit(ohe)
cluster_assignment = clustering_model.labels_

articles = df["Article"]

article_clusters = [[] for i in range(num_clusters)]
for article_id, cluster_id in enumerate(cluster_assignment):
    article_clusters[cluster_id].append(articles[article_id])

for i, cluster in enumerate(article_clusters):
    print("Cluster ", i+1)
    #print(cluster)
    print("")

print(article_clusters[1][0])
print("\n\n\n")
print(article_clusters[1][1])
print("\n\n\n")
print(article_clusters[1][2])
print("\n\n\n")
print(article_clusters[1][3])
