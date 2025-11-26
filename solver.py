import pandas as pd
import hashlib
import numpy as np
import shap
import nltk
nltk.download('punkt_tab')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

#########################################
# CONFIG
#########################################

STUDENT_ID = "STU039"

BOOKS_FILE = "books.csv"
REVIEWS_FILE = "reviews.csv"

#########################################
# STEP 1 — FIND THE MANIPULATED BOOK
#########################################

def get_student_hash(student_id):
    h = hashlib.sha256(student_id.encode()).hexdigest()[:8].upper()
    return h


def find_target_book(books, reviews, student_hash):
    
    # I tried to filter the books with rating_number=1234 & avg_rating=5.0
    target_books = books[
        (books["rating_number"] == 1234) &
        (books["average_rating"] == 5.0)
    ]

    if target_books.empty:
        raise Exception("No books found with rating_number=1234 and avg_rating=5.0")

    # then checked the reviews for hash in text
    mask = reviews["text"].str.contains(student_hash, case=False, na=False)
    fake_review = reviews[mask]

    if fake_review.empty:
        raise Exception("No review contains your hash!")

    # Find ASIN of fake review
    print(books.columns)

    print(reviews.columns)
    asin = fake_review.iloc[0]["asin"]

    # Finding the corresponding book
    book = target_books[target_books["parent_asin"] == asin]
    if book.empty:
        raise Exception("Fake review found but book does not match filtering rules!")

    return book.iloc[0], fake_review.iloc[0]


def compute_flag1(book_title):
    
    # Taking the first 8 non-space characters
    key = "".join(book_title.split())[:8]
    return hashlib.sha256(key.encode()).hexdigest()


#########################################
# STEP 2 — FLAG2
#########################################

def compute_flag2(student_hash):
    return f"FLAG2{{{student_hash}}}"


#########################################
# STEP 3 — FLAG3 USING SHAP
#########################################

def label_suspicion(row):
    text = row["text"]
    rating = row["rating"]

    if rating < 5:
        return 0

    words = text.split()
    short = len(words) < 10
    super_words = ["amazing", "awesome", "best", "incredible", "unbelievable", "mustread"]

    has_super = any(w.lower() in text.lower() for w in super_words)

    return 1 if short or has_super else 0


def compute_flag3(reviews, book_asin, student_id):
    # Extracting all reviews for the book
    book_reviews = reviews[reviews["asin"] == book_asin].copy()

    # Label data
    book_reviews["label"] = book_reviews.apply(label_suspicion, axis=1)

    # Vectorize text
    vectorizer = TfidfVectorizer(stop_words="english", min_df=2)
    X = vectorizer.fit_transform(book_reviews["text"])
    y = book_reviews["label"]

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    # Predict suspicion scores
    scores = model.predict_proba(X)[:, 1]

    # Choose genuine reviews (low suspicion)
    genuine_mask = scores < 0.3
    X_genuine = X[genuine_mask]

    # SHAP analysis
    explainer = shap.LinearExplainer(model, X, feature_names=vectorizer.get_feature_names_out())
    shap_values = explainer.shap_values(X_genuine)

    # Mean SHAP contribution per word
    mean_shap = shap_values.mean(axis=0)
    features = vectorizer.get_feature_names_out()

    # Words most negative → reduce suspicion
    top3_idx = np.argsort(mean_shap)[:3]
    top3_words = [features[i] for i in top3_idx]

    # Prepare FLAG3 string
    numeric_id = ''.join(filter(str.isdigit, student_id))
    s = "".join(top3_words) + numeric_id
    digest = hashlib.sha256(s.encode()).hexdigest()[:10]

    return f"FLAG3{{{digest}}}", top3_words


#########################################
# MAIN EXECUTION
#########################################

def main():
    student_hash = get_student_hash(STUDENT_ID)

    print("Reading CSV files...")
    books = pd.read_csv(BOOKS_FILE)
    reviews = pd.read_csv(REVIEWS_FILE)

    print("Finding manipulated book...")
    book, fake_review = find_target_book(books, reviews, student_hash)

    print("Computing FLAG1...")
    FLAG1 = compute_flag1(book["title"])

    print("Computing FLAG2...")
    FLAG2 = compute_flag2(student_hash)

    print("Computing FLAG3 (running SHAP)...")
    FLAG3, top_words = compute_flag3(reviews, book["parent_asin"], STUDENT_ID)

    print("\n======== RESULTS ========")
    print("Book Title:", book["title"])
    print("Fake Review Text:", fake_review["text"])
    print("Top Genuine Words:", top_words)
    print("\nFLAG1 =", FLAG1)
    print("FLAG2 =", FLAG2)
    print("FLAG3 =", FLAG3)


if __name__ == "__main__":
    main()

