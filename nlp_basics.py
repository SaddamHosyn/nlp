import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
    



#Step 2: Text Preprocessing

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Raw text input
text = "The quick brown fox jumps over the lazy dog!"

# Tokenization
tokens = word_tokenize(text.lower())

# Remove punctuation and stop words
stop_words = set(stopwords.words('english'))
# curious what words are actually in there.
print("all output:", stopwords.words('english'))
# Output: ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", ...]

filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]

print("filtered output:", filtered_tokens)
# Output: ['quick', 'brown', 'fox', 'jumps', 'lazy', 'dog']




# Step 3: Syntactic Parsing and Analysis
# Load English language model
nlp = spacy.load("en_core_web_sm")

# Process text
doc = nlp("The cat sat on the mat")

# Part-of-speech tagging
for token in doc:
    print(f"{token.text}: {token.pos_}")

# Output:
# The: DET
# cat: NOUN
# sat: VERB
# on: ADP
# the: DET
# mat: NOUN
