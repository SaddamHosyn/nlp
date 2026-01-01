import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from transformers import MarianMTModel, MarianTokenizer
from sklearn.metrics import classification_report


print("\n--- [START] Script Starting ---")


#Step 2: Text Preprocessing

#  Download required NLTK data. usually we dont write code like this in the actaul dvelopeemnt settings, but for the sake of completeness of this example script we include it here.
print("DEBUG: Checking NLTK data (punkt, stopwords)...")
nltk.download('punkt')
nltk.download('stopwords')
print("DEBUG: NLTK data check complete.")

# Raw text input
text = "The quick brown fox jumps over the lazy dog!"
print(f"DEBUG: Original Text: '{text}'")

# Tokenization
tokens = word_tokenize(text.lower())
print(f"DEBUG: Tokenized List: {tokens}")

# Remove punctuation and stop words
stop_words = set(stopwords.words('english'))
print(f"DEBUG: Loaded {len(stop_words)} stopwords from NLTK.")
# print("all output:", stopwords.words('english')) # Optional: print all stopwords
# Output: ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", ...]

filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]

print("filtered output:", filtered_tokens)
# Output: ['quick', 'brown', 'fox', 'jumps', 'lazy', 'dog']



# Step 3: Syntactic Parsing and Analysis
print("\n--- [STEP 3] Syntactic Parsing & Analysis ---")

# Load English language model
print("DEBUG: Loading spaCy model 'en_core_web_sm'...")
nlp = spacy.load("en_core_web_sm")
print("DEBUG: spaCy model loaded.")

# Process text
sent_parse = "The cat sat on the mat"
doc = nlp(sent_parse)
print(f"DEBUG: Analyzing sentence: '{sent_parse}'")

# Part-of-speech tagging
print("DEBUG: Part-of-Speech Tags:")
for token in doc:
    print(f"  - {token.text}: {token.pos_}")


# Step 4: Feature Engineering and Text Representation
print("\n--- [STEP 4] Feature Engineering (Embeddings) ---")

# Load pre-trained model
print("DEBUG: Loading SentenceTransformer 'all-MiniLM-L6-v2'...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("DEBUG: SentenceTransformer model loaded.")

# Convert sentences to embeddings
sentences = ["The cat sits on the mat", "The feline rests on the rug"]
print(f"DEBUG: Encoding {len(sentences)} sentences...")
embeddings = model.encode(sentences)

print(f"Embedding shape: {embeddings.shape}")
print("DEBUG: First 5 values of first embedding:", embeddings[0][:5]) # Debugging peek
# print(embeddings) # Uncomment to see full wall of numbers


# Step 5: Modeling and Pattern Recognition
print("\n--- [STEP 5] Modeling (Sentiment Analysis) ---")

# Load a pre-trained sentiment analysis model
print("DEBUG: Loading Sentiment Analysis pipeline...")
classifier = pipeline("sentiment-analysis")
print("DEBUG: Pipeline loaded.")

# Classify text sentiment
texts = ["I love this product!", "This is terrible and disappointing"]
print(f"DEBUG: Analyzing sentiment for: {texts}")
results = classifier(texts)

print("DEBUG: Classification Results:")
for text, result in zip(texts, results):
    print(f"Text: {text}")
    print(f"Sentiment: {result['label']}, Confidence: {result['score']:.2f}\n")

# Extra check
test_sent = "I love this product! It works great."
print(f"DEBUG: Quick check for '{test_sent}':")
print(classifier(test_sent))


# Step 6: Evaluation and Deployment
print("\n--- [STEP 6] Evaluation (Metrics) ---")

# Example predictions vs actual labels
y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1]
print("DEBUG: Calculating classification report for dummy data...")
print(f"DEBUG: y_true={y_true}, y_pred={y_pred}")

# Generate evaluation metrics
print("Report Output:")
print(classification_report(y_true, y_pred))


# Step 7: Translation Task (Bonus)
print("\n--- [STEP 7] Translation Task ---")

# Load translation model
model_name = 'Helsinki-NLP/opus-mt-en-es'
print(f"DEBUG: Loading translation model '{model_name}'...")


#from transformers import AutoModelForSeq2SeqLM
#model = AutoModelForSeq2SeqLM.from_pretrained('Helsinki-NLP/opus-mt-en-es')
#Beginner: Use MarianMTModel (Good for learning).

#Pro: Use AutoModelForSeq2SeqLM (Flexible and robust).

tokenizer = MarianTokenizer.from_pretrained(model_name)
model_trans = MarianMTModel.from_pretrained(model_name) # Renamed to avoid variable clash
print("DEBUG: Translation model & tokenizer loaded.")

# Translate English to Spanish (Note: Model name says en-es, output likely Spanish not German)
text_to_trans = "Hello, how are you?"
print(f"DEBUG: Translating '{text_to_trans}'...")
translated = model_trans.generate(**tokenizer(text_to_trans, return_tensors="pt", padding=True))
translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)

print(f"Original: {text_to_trans}")
print(f"Translated: {translated_text}")

print("\n--- [END] Script Finished Successfully ---")
