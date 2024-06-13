import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    movie = pd.read_csv("C:/Users/goyal/OneDrive/Documents/dataset.csv")
except FileNotFoundError:
    print("File not found. Please check the file path.")

movie['genre'] = movie['genre'].astype(str)
movie['overview'] = movie['overview'].astype(str)

movie['tag'] = movie['genre'] + movie['overview']
movie['1_title'] = movie['title'].astype(str).str.lower()
dataset1 = movie[['id', '1_title', 'tag']]

cv = CountVectorizer(max_features=10000, stop_words='english')
vec = cv.fit_transform(dataset1['tag'].values.astype('U')).toarray()
sim = cosine_similarity(vec)

def recommend(title):
    title = title.lower()
    try:
        indices = dataset1[dataset1['1_title'] == title].index
        if len(indices) == 0:
            print("Movie not found in the dataset")
        else:
            index = indices[0]
            dist = sorted(list(enumerate(sim[index])), reverse=True, key=lambda vec: vec[1])
            for i in dist[1:6]:
                print(dataset1.iloc[i[0]]['1_title'])
    except Exception as e:
        print("An error occurred: ", str(e))

Movie_name = input("Enter movie: ")
recommend(Movie_name)