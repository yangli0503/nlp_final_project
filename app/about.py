import streamlit as st

st.markdown("# About the Dashboard")
st.html('''

<h2>NLP Final Project</h2>

<h4>Project Title: Song Lyric Classification</h4>

<h4>Team Names: Steph Clampitt, Anna Graves, Kierra Willis</h4>

<h4>Project Description:</h4> 

<p>The primary goal of this project is to practice text classification on real world Data. 
We plan on applying this NLP method to song lyrics to classify them based on their meaning. 
Our process will include Data cleaning (for example, removing stop words and special characters). 
We will then vectorize the dataset using the TfidfVectorizer.</p>

<p>Our dataset contains the top 100 songs with their lyrics each year from 1959 to 2019. 
Because we do not have labels, we will use text clustering to reveal patterns in the lyrics. 
The primary NLP methods we will focus on are K-Means and LDA. K-Means will help us see clear and 
easy to interpret clusters, and LDA will help us see common topics among the songs’ lyrics. After 
we apply these techniques, we will visualize our findings with Python and a Python framework like 
Django, Flask or Streamlit.</p>
''')

st.markdown("## References")
st.html('''
Dataset: https://www.kaggle.com/datasets/brianblakely/top-100-songs-and-lyrics-from-1959-to-2019
''')