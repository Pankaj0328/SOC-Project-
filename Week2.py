import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Text preprocessing
text = "Natural Language Processing is AMAZING! It helps computers understand text."
clean_text = re.sub(r'[^\w\s]', '', text.lower())
print("Cleaned:", clean_text)

# Tokenization
tokens = word_tokenize(clean_text)
stop_words = set(stopwords.words('english'))
filtered_tokens = [w for w in tokens if w not in stop_words]
print("Tokens:", filtered_tokens)

# TF-IDF
documents = [
    "machine learning is powerful",
    "natural language processing helps computers",
    "text analysis requires preprocessing"
]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)
print("Features:", vectorizer.get_feature_names_out())

# Basic sentiment analysis
sentiment_data = [
    ("I love this product", "positive"),
    ("Terrible quality", "negative"),
    ("Good value for money", "positive"),
    ("Poor service", "negative")
]

texts, labels = zip(*sentiment_data)
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.5)

tfidf = TfidfVectorizer()
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

classifier = MultinomialNB()
classifier.fit(X_train_vec, y_train)
predictions = classifier.predict(X_test_vec)
print("Predictions:", predictions)
