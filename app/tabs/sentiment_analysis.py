from Data.data import get_preprocessed_data
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from pathlib import Path
import joblib
import pyLDAvis
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import nltk
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning, module='jupyter_client')

st.header("Sentiment Analysis Visualizations")


#using vader: a sentiment analysis tool used specifically for social media and lyrics (and it works on unlabeled data)
from nltk.sentiment import vader

#analyer tool that tokenizes and returns scores
analyzertool = vader.SentimentIntensityAnalyzer()

df_clean = get_preprocessed_data()

#applying the analyzer to each lyric
sent_cols = df_clean['lyrics'].apply(analyzertool.polarity_scores).apply(pd.Series)
df_clean = pd.concat([df_clean, sent_cols], axis=1)

#label using common VADER thresholds
def label_from_compound(x, pos_thr=0.4, neg_thr=-0.4):
    if x >= pos_thr: return 'pos'
    if x <= neg_thr: return 'neg'
    return 'neu'

df_clean['sentiment_label'] = df_clean['compound'].apply(label_from_compound)

#graphs for sentiment analysis
# --- Yearly average sentiment (table) ---
yearly = (
    df_clean.groupby("year", as_index=False)["compound"]
    .mean()
    .sort_values("year")
)

st.subheader("Yearly Average Sentiment (VADER compound)")
st.dataframe(yearly)

# --- Yearly average sentiment (line chart) ---
st.subheader("Average Sentiment of Top-10 Songs by Year (>= 2000)")
fig, ax = plt.subplots()
ax.plot(yearly["year"], yearly["compound"])
ax.set_xlabel("Year")
ax.set_ylabel("Mean VADER compound")
ax.set_title("Average Sentiment of Top-10 Songs by Year (>= 2000)")
st.pyplot(fig)   # <-- instead of plt.show()

# --- Top & bottom songs by compound ---
top5 = df_clean.nlargest(5, "compound")[["year", "artist", "song", "compound"]]
bottom5 = df_clean.nsmallest(5, "compound")[["year", "artist", "song", "compound"]]

st.subheader("Top 5 Most Positive Songs")
st.dataframe(top5)

st.subheader("Top 5 Most Negative Songs")
st.dataframe(bottom5)