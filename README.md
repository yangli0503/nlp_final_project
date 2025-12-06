NLP Final Project

Project Title:  Song Lyric Classification 

Team Names: Steph Clampitt, Anna Graves, Kierra Willis 

Project Description: The primary goal of this project is to practice text classification on real world data. We plan on applying this NLP method to song lyrics to classify them based on their meaning. Our process will include data cleaning (for example, removing stop words and special characters).  We will then vectorize the dataset using the TfidfVectorizer. 

Our dataset contains the top 100 songs with their lyrics each year from 1959 to 2019. Because we do not have labels, we will use text clustering to reveal patterns in the lyrics. The primary NLP methods we will focus on are K-Means and LDA. K-Means will help us see clear and easy to interpret clusters, and LDA will help us see common topics among the songs’ lyrics. After we apply these techniques, we will visualize our findings with Python and a Python framework like Django, Flask or Streamlit. 

Dataset: https://www.kaggle.com/datasets/brianblakely/top-100-songs-and-lyrics-from-1959-to-2019 

Note to run for development:

Run this in your terminal/console to run the application.
streamlit run /~/nlp_final_project/app/app.py

