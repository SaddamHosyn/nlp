import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
from sentence_transformers import SentenceTransformer
from transformers import pipeline

from sklearn.metrics import classification_report



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


# Step 4: Feature Engineering and Text Representation


# Load pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Convert sentences to embeddings
sentences = ["The cat sits on the mat", "The feline rests on the rug"]
embeddings = model.encode(sentences)

print(f"Embedding shape: {embeddings.shape}")
# Output: Embedding shape: (2, 384)  
print(embeddings)

 
 # Step 5: Modeling and Pattern Recognition



# Load a pre-trained sentiment analysis model
classifier = pipeline("sentiment-analysis")

# Classify text sentiment
texts = ["I love this product!", "This is terrible and disappointing"]
results = classifier(texts)

for text, result in zip(texts, results):
    print(f"Text: {text}")
    print(f"Sentiment: {result['label']}, Confidence: {result['score']:.2f}\n")

# Output:
# Text: I love this product!
# Sentiment: POSITIVE, Confidence: 0.99
#
# Text: This is terrible and disappointing
# Sentiment: NEGATIVE, Confidence: 0.99


 # Step 6: Evaluation and Deployment




# Example predictions vs actual labels
y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1]

# Generate evaluation metrics
print(classification_report(y_true, y_pred))




#   Sentiment analysis and text classification
from transformers import pipeline

# Load sentiment analysis pipeline
classifier = pipeline("sentiment-analysis")

# Analyze sentiment
result = classifier("I love this product! It works great.")
print(result)
# Output: [{'label': 'POSITIVE', 'score': 0.9998}]
