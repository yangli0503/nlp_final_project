import ssl
import spacy
from spacy.cli import download

def download_spacy_model():
    # Fix SSL issues on Mac
    try:
        _create_unverified_https_context = ssl._create_unverified_context
        ssl._create_default_https_context = _create_unverified_https_context
    except Exception:
        pass

    # Check whether spaCy model already exists
    try:
        spacy.load("en_core_web_sm")
        print("spaCy model already installed.")
    except OSError:
        print("Downloading spaCy en_core_web_sm...")
        download("en_core_web_sm")
        print("Download complete.")
