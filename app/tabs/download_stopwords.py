# app/tabs/download_stopwords.py

import nltk
from pathlib import Path

# use the same local nltk_data folder as your other resources
NLTK_DIR = Path(__file__).parent / "nltk_data"
NLTK_DIR.mkdir(exist_ok=True)

# make sure NLTK knows to look here
nltk.data.path.append(str(NLTK_DIR))

# download stopwords INTO this folder
nltk.download("stopwords", download_dir=str(NLTK_DIR))
print("Downloaded stopwords to:", NLTK_DIR)
