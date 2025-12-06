import spacy
import pandas as pd
import re
import os

def cleaning(doc):
  #Lemmatizes and removes stopwords
  txt = [token.lemma_ for token in doc if not token.is_stop]
  #only keep sentences with at least 3 words
  if len(txt) > 2:
    return ' '.join(txt)


def get_preprocessed_data():
  print(os.getcwd())

  filename = "Data/preprocessed_data.csv"

  if os.path.exists(filename):
    file = open(filename, "r")
    df = pd.read_csv(file)
    return df

  else:
    nlp = spacy.load('en_core_web_sm')
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv('Data/all_songs_data.csv')

    # Convert the 'Year' column to numeric, coercing errors to NaN
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

    # Drop rows where 'Year' could not be converted to a number (NaN)
    df.dropna(subset=['Year'], inplace=True)

    #Keep top 10 songs from last 25 years
    df = df[(df['Year'] >= 2000) & (df['Rank'] <= 10)]

    #Replace n-word with n-word
    df['Lyrics'] = df['Lyrics'].str.replace('nigga', 'nword')

    brief_cleaning = (re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in df['Lyrics'])

    #Use spacy pipeline to clean the Lyrics columns
    txt = [cleaning(doc) for doc in nlp.pipe(brief_cleaning, batch_size=5000, n_process=-1)]

    #Create finished dataset with artist, song title, rank, year, and clean lyrics
    df_clean = pd.DataFrame({'artist': df['Artist'], 'song': df['Song Title'], 'rank': df['Rank'], 'year': df['Year'], 'lyrics': txt})

    #There are 2 top 10 songs that don't have lyrics - (1) Maria Maria by Santana and (2) Can't Hold Us by Macklemore - These will be dropped
    df_clean = df_clean.dropna().copy()

    df_clean.to_csv(filename, index=False)
    return df_clean