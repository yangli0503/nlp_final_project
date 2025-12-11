import ssl
import nltk

# --- Fix SSL issue on Mac so NLTK can download things ---
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('vader_lexicon')
print("Done downloading vader_lexicon.")

import ssl
import nltk

# Fix SSL issue on Mac so NLTK can download files
try:
    _create_unverified_https_context = ssl._create_unverified_context
    ssl._create_default_https_context = _create_unverified_https_context
except Exception:
    pass

nltk.download('vader_lexicon')

print("VADER lexicon downloaded successfully.")
