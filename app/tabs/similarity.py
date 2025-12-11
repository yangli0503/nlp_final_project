from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from Data.data import get_preprocessed_data
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import streamlit as st
from nltk.corpus import stopwords
import matplotlib.colors as mcolors

st.header("Song Similarity")

df_clean = get_preprocessed_data()
nltk.data.path.append("nltk_data")

stop_words = stopwords.words("english")

#Add additional stopwords - mostly filler words in songs
stop_words_custom = ['yeah', 'ya', 'yea', 'oh', 'ohh', 'ooh', 'woah', 'whoa', 'ayy', 'uh', 'na', 'hey', 'la', 'doo', 'da']
# Combine the stop words lists and convert to a set to ensure uniqueness, then back to a list
combined_stop_words = list(set(stop_words + stop_words_custom))

vectorizer = TfidfVectorizer(stop_words=combined_stop_words)
vectorized_songs = vectorizer.fit_transform(df_clean['lyrics'])

#Get number of coments that explain 95% of variance
pca = PCA(n_components=0.95)
reduced_data = pca.fit_transform(vectorized_songs.toarray())

num_clusters = 5
km = KMeans(n_clusters=num_clusters, max_iter=10000, n_init=50, random_state=42).fit(vectorized_songs)

df_clean['kmeans_cluster'] = km.labels_

song_clusters = (df_clean[['song', 'kmeans_cluster', 'rank']]
                 .sort_values(by=['kmeans_cluster', 'rank'])
                 .groupby('kmeans_cluster').head(20))
song_clusters = song_clusters.copy(deep=True)

# Get feature names from the TfidfVectorizer used for clustering
feature_names = vectorizer.get_feature_names_out()
topn_features = 15
ordered_centroids = km.cluster_centers_.argsort()[:, ::-1]

# get key features for each cluster
# get movies belonging to each cluster

cluster_colors = {}

# ----- Build the cluster colors from matplotlib scatter -----
for i in range(num_clusters):
    points = plt.scatter(
        reduced_data[km.labels_ == i, 0],
        reduced_data[km.labels_ == i, 1],
        label=i
    )
    cluster_colors[i] = points.get_facecolor()[0]   # RGBA tuple


plt.legend()
st.pyplot(plt.gcf())

# ----- Create columns -----
cols = st.columns(num_clusters)

for i in range(num_clusters):
    with cols[i]:
        hex_color = mcolors.to_hex(cluster_colors[i])

        key_features = [
            feature_names[index]
            for index in ordered_centroids[i, :topn_features]
        ]

        songs = song_clusters[
            song_clusters['kmeans_cluster'] == i
        ]['song'].values.tolist()

        st.html(f"""
            <h3 style="color: {hex_color};">
                Cluster #{i+1}
            </h3>
        """)

        for feature in key_features:
            st.write(f"- {feature}")

        # ----- Dropdown (Expander) for Songs -----
        with st.expander("Popular Songs", expanded=False):
            for song in songs:
                st.write(f"{song}")

