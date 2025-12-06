from Data.data import get_preprocessed_data
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from pathlib import Path
import joblib
import pyLDAvis
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning, module='jupyter_client')

# Text to numbers and make document-term matrix
df_clean = get_preprocessed_data()
vectorizer = CountVectorizer(
    max_df=.5,
    min_df=5,
    stop_words='english',
    max_features=2000)
dtm = vectorizer.fit_transform(df_clean['lyrics'])

# Create LDA model
lda_all = LatentDirichletAllocation(
    n_components=5,
    max_iter=500,
    learning_method='batch',
    evaluate_every=10,
    random_state=42,
    verbose=1)
# Train it
lda_all.fit(dtm)

topic_model_path = Path('Data', 'topics')
if not topic_model_path.exists():
    topic_model_path.mkdir(exist_ok=True, parents=True)

lda_file = topic_model_path / 'lda_all.pkl'

# Only save if it doesn't exist
if not lda_file.exists():
    joblib.dump(lda_all, lda_file)
    print("Model saved.")
else:
    print("File already exists, skipping save.")

lda_all = joblib.load(topic_model_path / 'lda_all.pkl')

st.header("Interactive Topic Visualization")

# topic-term distribution
topic_term_file = topic_model_path / 'topic_term_dists.pkl'
if topic_term_file.exists():
    topic_term_dists = joblib.load(topic_term_file)
else:
    topic_term_dists = lda_all.components_ / lda_all.components_.sum(axis=1)[:, None]
    joblib.dump(topic_term_dists, topic_term_file)

# document-topic distribution
doc_topic_file = topic_model_path / 'doc_topic_dists.pkl'
if doc_topic_file.exists():
    doc_topic_dists = joblib.load(doc_topic_file)
else:
    doc_topic_dists = lda_all.transform(dtm)
    joblib.dump(doc_topic_dists, doc_topic_file)

# document lengths (sum of tokens in each document)
doc_lengths_file = topic_model_path / 'doc_lengths.pkl'
if doc_lengths_file.exists():
    doc_lengths = joblib.load(doc_lengths_file)
else:
    doc_lengths = np.asarray(dtm.sum(axis=1)).reshape(-1)
    joblib.dump(doc_lengths, doc_lengths_file)

# vocabulary
vocab_file = topic_model_path / 'vocab.pkl'
if vocab_file.exists():
    vocab = joblib.load(vocab_file)
else:
    vocab = vectorizer.get_feature_names_out()
    joblib.dump(vocab, vocab_file)

# term frequencies (total word counts across all docs)
term_frequency_file = topic_model_path / 'term_frequency.pkl'
if term_frequency_file.exists():
    term_frequency = joblib.load(term_frequency_file)
else:
    term_frequency = np.asarray(dtm.sum(axis=0)).reshape(-1)
    joblib.dump(term_frequency, term_frequency_file)

# --- Prepare visualization ---
vis_data = pyLDAvis.prepare(
    topic_term_dists,
    doc_topic_dists,
    doc_lengths,
    vocab,
    term_frequency
)

html_string = pyLDAvis.prepared_data_to_html(vis_data)
style_fixer = """
<style>
    .st-emotion-cache-1w723zb {
        max-width: 1250px !important;
    }
</style>
"""
components.html(html_string, height=900, scrolling=True)
st.html(style_fixer)




