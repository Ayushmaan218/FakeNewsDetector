import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier # Import the new classifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
import joblib
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# --- Download NLTK data (only needs to be done once) ---
try:
    stopwords.words('english')
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')
    print("Download complete.")

# --- Define file paths ---
DATA_DIR = 'data'
MODEL_DIR = 'model'
FAKE_CSV = os.path.join(DATA_DIR, 'Fake.csv')
REAL_CSV = os.path.join(DATA_DIR, 'True.csv')
MODEL_FILENAME = os.path.join(MODEL_DIR, 'fake_news_model.joblib')

# --- Text Preprocessing Function ---
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Cleans the input text by lowercasing, removing punctuation and numbers,
    removing stopwords, and applying stemming.
    """
    if not isinstance(text, str):
        text = str(text)
        
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    text = re.sub(r'\d+', '', text)      # remove numbers
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

# --- 1. Load and Prepare Datasets ---
try:
    print("--- Loading Datasets ---")
    df_fake = pd.read_csv(FAKE_CSV, encoding='utf-8')
    df_real = pd.read_csv(REAL_CSV, encoding='utf-8')
    print("Datasets loaded successfully.")

    df_fake['label'] = 1
    df_real['label'] = 0
    df_combined = pd.concat([df_real, df_fake], ignore_index=True)

    # Combine title and text
    df_combined['content'] = df_combined['title'].fillna('') + ' ' + df_combined['text'].fillna('')
    df_combined.dropna(subset=['content'], inplace=True)
    df_combined = df_combined[df_combined['content'].str.strip() != '']

    # Apply the cleaning function
    print("\n--- Cleaning and Preprocessing Text (this may take a while) ---")
    df_combined['cleaned_content'] = df_combined['content'].apply(clean_text)
    print("Text cleaning complete.")
    
    df = shuffle(df_combined, random_state=42).reset_index(drop=True)
    
    print("\n--- Combined and Cleaned Data Sample ---")
    print(df[['cleaned_content', 'label']].head())
    print(f"\nTotal articles: {len(df)}")
    print("\n" + "="*30 + "\n")

except FileNotFoundError:
    print(f"!!! ERROR: Make sure '{FAKE_CSV}' and '{REAL_CSV}' exist in the 'data' folder. !!!")
    exit()

# --- 2. Data Preparation ---
X = df['cleaned_content']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- 3. Model Pipeline Construction ---
print("--- Building the Model Pipeline with PassiveAggressiveClassifier ---")
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
    # Swapped LogisticRegression with the new, more powerful classifier
    ('classifier', PassiveAggressiveClassifier(max_iter=50, random_state=42)) 
])
print("Pipeline created successfully.\n")

# --- 4. Training the Model ---
print("--- Training the Model ---")
pipeline.fit(X_train, y_train)
print("Model training complete.\n")

# --- 5. Save the Trained Model ---
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(pipeline, MODEL_FILENAME)
print(f"--- Model saved to {MODEL_FILENAME} ---")
print("\n" + "="*30 + "\n")

# --- 6. Evaluating the Model ---
print("--- Evaluating the Model on the Test Set ---")
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Real News (0)', 'Fake News (1)'])
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(report)
