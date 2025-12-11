import ssl
import nltk

def download_vader():
    # Fix SSL issue on Mac so NLTK can download things
    try:
        _create_unverified_https_context = ssl._create_unverified_context
        ssl._create_default_https_context = _create_unverified_https_context
    except Exception:
        pass

    # Only download if not present
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
        print("VADER lexicon already installed.")
    except LookupError:
        nltk.download("vader_lexicon")
        print("VADER lexicon downloaded successfully.")
