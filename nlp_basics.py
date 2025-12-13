import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Raw text input
text = "The quick brown fox jumps over the lazy dog!"

# Tokenization
tokens = word_tokenize(text.lower())

# Remove punctuation and stop words
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]

print(filtered_tokens)
# Output: ['quick', 'brown', 'fox', 'jumps', 'lazy', 'dog']
