import streamlit as st
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

overview_page = st.Page('about.py', title='About')

dash_pages = [
    st.Page('tabs/topics.py', title = 'Topics'),
    st.Page('tabs/sentiment_analysis.py', title = 'Sentiment Analysis'),
    st.Page('tabs/similarity.py', title = 'Similarity'),
    st.Page('tabs/dendrogram.py', title = 'Dendrogram'),
]

page_navigation = st.navigation({
    'About': [overview_page],
    'Dashboard': dash_pages,
})

page_navigation.run()

