import ssl
import spacy

# Fix SSL issues on Mac for model download
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download the small English model
import subprocess

# Fix SSL error on Mac
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except Exception:
    pass

subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
