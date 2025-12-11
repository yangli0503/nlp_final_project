from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from Data.data import get_preprocessed_data
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import streamlit as st
from scipy.cluster.hierarchy import dendrogram, linkage

st.header("Dendrogram")

def plot_hierarchical_clusters(linkage_matrix, data, p=100, figure_size=(8,12)):
    # set size
    fig, ax = plt.subplots(figsize=figure_size)
    movie_titles = data['song'].values.tolist()
    # plot dendrogram
    R = dendrogram(linkage_matrix, orientation="left", labels=movie_titles,
                    truncate_mode='lastp',
                    p=p,
                    no_plot=True)
    temp = {R["leaves"][ii]: movie_titles[ii] for ii in range(len(R["leaves"]))}
    def llf(xx):
        return "{}".format(temp[xx])
    ax = dendrogram(
            linkage_matrix,
            truncate_mode='lastp',
            orientation="left",
            p=p,
            leaf_label_func=llf,
            leaf_font_size=10.,
            )
    plt.tick_params(axis= 'x',
                    which='both',
                    bottom='off',
                    top='off',
                    labelbottom='off')
    plt.tight_layout()
    st.pyplot(plt.gcf())

df_clean = get_preprocessed_data()
stop_words = nltk.corpus.stopwords.words('english')

#Add additional stopwords - mostly filler words in songs
stop_words_custom = ['yeah', 'ya', 'yea', 'oh', 'ohh', 'ooh', 'woah', 'whoa', 'ayy', 'uh', 'na', 'hey', 'la', 'doo', 'da']
# Combine the stop words lists and convert to a set to ensure uniqueness, then back to a list
combined_stop_words = list(set(stop_words + stop_words_custom))

vectorizer = TfidfVectorizer(stop_words=combined_stop_words)
vectorized_songs = vectorizer.fit_transform(df_clean['lyrics'])

model_matrix = vectorized_songs.toarray()

similarity_matrix = cosine_similarity(model_matrix)

# Generate the linkage matrix
# Using 'ward' method for linkage, but other methods can be used
Z = linkage(model_matrix, 'ward')

plot_hierarchical_clusters(Z, p=100, data=df_clean, figure_size=(20,20))
