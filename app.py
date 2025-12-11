import streamlit as st
import sys
from pathlib import Path
from download_vader import download_vader
from download_spacy_model import download_spacy_model

@st.cache_resource
def setup_once():
    download_vader()
    # download_spacy_model()

setup_once()   # runs ONLY the first time

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

overview_page = st.Page('app/about.py', title='About')

dash_pages = [
    st.Page('app/tabs/topics.py', title ='Topics'),
    st.Page('app/tabs/sentiment_analysis.py', title ='Sentiment Analysis'),
    st.Page('app/tabs/similarity.py', title ='Similarity'),
    st.Page('app/tabs/dendrogram.py', title ='Dendrogram'),
]

page_navigation = st.navigation({
    'About': [overview_page],
    'Dashboard': dash_pages,
})

page_navigation.run()

