from Data.data import get_preprocessed_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from pathlib import Path
import joblib
import pyLDAvis
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import warnings

warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning, module='jupyter_client')


df_clean = get_preprocessed_data()

st.header("Similarity Visualization")

# ---- STOPWORDS (no NLTK) ----
# base English stopwords from scikit-learn
base_stop_words = set(text.ENGLISH_STOP_WORDS)

# your custom song filler words
stop_words_custom = {'yeah', 'yea', 'oh', 'ohh', 'woah', 'ayy', 'uh', 'na', 'hey'}

# combine them
combined_stop_words = list(base_stop_words.union(stop_words_custom))

# TF-IDF with combined stopwords
vectorizer = TfidfVectorizer(stop_words=combined_stop_words)
vectorized_songs = vectorizer.fit_transform(df_clean['lyrics'])



import prince
famd = prince.FAMD(
    n_components=10,
    n_iter=10,
    copy=True,
    check_input=True,
    random_state=42,
    engine="sklearn",
    handle_unknown="error"
)
famd = famd.fit(df_clean)
famd.eigenvalues_summary

famd.plot(
    df_clean,
    x_component=0,
    y_component=1
)

from sklearn.cluster import KMeans

row_coords = famd.row_coordinates(df_clean)
famd_10 = row_coords.iloc[:, :13]

wcss=[]

for i in range(1,15):
     kmeans = KMeans(n_clusters=i, init ='k-means++', max_iter=300,  n_init=15,random_state=0 )
     kmeans.fit(famd_10)
     wcss.append(kmeans.inertia_)

plt.plot(np.arange(1,15),wcss) # range we chose for "for loop"
plt.title('The Elbow Method Graph')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
reduced_data = pca.fit_transform(vectorized_songs.toarray())


num_clusters = 5
km = KMeans(n_clusters=num_clusters, max_iter=10000, n_init=50, random_state=42).fit(vectorized_songs)

df_clean['kmeans_cluster'] = km.labels_

song_clusters = (df_clean[['song', 'kmeans_cluster', 'rank']]
                 .sort_values(by=['kmeans_cluster', 'rank'])
                 .groupby('kmeans_cluster').head(20))
song_clusters = song_clusters.copy(deep=True)
song_clusters

# Get feature names from the TfidfVectorizer used for clustering
feature_names = vectorizer.get_feature_names_out()
topn_features = 15
ordered_centroids = km.cluster_centers_.argsort()[:, ::-1]

# get key features for each cluster
# get movies belonging to each cluster
for cluster_num in range(num_clusters):
    key_features = [feature_names[index]
                        for index in ordered_centroids[cluster_num, :topn_features]]
    songs = song_clusters[song_clusters['kmeans_cluster'] == cluster_num]['song'].values.tolist()
    print('CLUSTER #'+str(cluster_num+1))
    print('Key Features:', key_features)
    print('Popular Songs:', songs)
    print('-'*80)

# 2D PCA coordinates corresponding to each song
# reduced_data shape: (n_songs, 2)

fig, ax = plt.subplots()

for i in range(num_clusters):
    ax.scatter(
        reduced_data[km.labels_ == i, 0],
        reduced_data[km.labels_ == i, 1],
        label=f"Cluster {i}"
    )

ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
ax.set_title("Song Clusters (TF-IDF + KMeans)")
ax.legend()

# Streamlit-friendly render
st.pyplot(fig)
