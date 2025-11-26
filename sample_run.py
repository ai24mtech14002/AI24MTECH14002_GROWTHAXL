import hashlib
import pandas as pd
import numpy as np
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import shap
import nltk
nltk.download('punkt')

BOOKS_CSV = "books.csv"     
REVIEWS_CSV = "reviews.csv"  

BOOK_ID_COL = "book_id"      # fallback if 'id' or 'bookId' not present
BOOK_TITLE_COL = "title"
BOOK_AVG_RATING_COL = "average_rating"   # could be 'average_rating' or 'averageRating'
BOOK_RATING_COUNT_COL = "rating_number"  # could be 'rating_count' or similar
REV_BOOK_ID_COL = "book_id"  # column in reviews that links to books
REV_TEXT_COL = "text"
REV_RATING_COL = "rating"    # numeric rating (e.g., 5)

# ---------- Step 0: Helper functions ----------
def sha256_hex(s: str):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def first_nonspace_chars(s: str, n=8):
    # remove spaces and take first n characters
    return "".join(s.split())[:n]

# ---------- Step 1: compute student hash ----------------
student_string = "STU160"   # EXACT string from the challenge
student_hash_full_hex = sha256_hex(student_string)
student_hash8 = student_hash_full_hex[:8].upper()
print("Student string:", student_string)
print("SHA256(full) (hex):", student_hash_full_hex)
print("Student HASH (first 8 hex, UPPER):", student_hash8)
# (This is the hash you will search for in reviews)

# ---------- Step 2: load datasets ----------
if not os.path.exists(BOOKS_CSV) or not os.path.exists(REVIEWS_CSV):
    raise FileNotFoundError(f"Make sure {BOOKS_CSV} and {REVIEWS_CSV} are present in the same folder.")

books = pd.read_csv(BOOKS_CSV, dtype=str)
reviews = pd.read_csv(REVIEWS_CSV, dtype=str)

# convert numeric columns where necessary
for c in [BOOK_AVG_RATING_COL, BOOK_RATING_COUNT_COL]:
    if c in books.columns:
        books[c] = pd.to_numeric(books[c], errors="coerce")

if REV_RATING_COL in reviews.columns:
    reviews[REV_RATING_COL] = pd.to_numeric(reviews[REV_RATING_COL], errors="coerce")

# If the book id column name different, try common alternatives
if BOOK_ID_COL not in books.columns:
    for alt in ["id", "bookId", "book_id"]:
        if alt in books.columns:
            BOOK_ID_COL = alt
            break
if REV_BOOK_ID_COL not in reviews.columns:
    for alt in ["book_id", "bookId", "bookId", "book"]:
        if alt in reviews.columns:
            REV_BOOK_ID_COL = alt
            break

print("Using book id column:", BOOK_ID_COL)
print("Using review -> book id column:", REV_BOOK_ID_COL)

# ---------- Step 3: find books with rating_number == 1234 and average_rating == 5.0 ----------
candidates = books[
    (books.get(BOOK_RATING_COUNT_COL) == 1234) |
    (books.get(BOOK_RATING_COUNT_COL) == "1234")
].copy()

# some datasets may label count column differently — try to coerce
if candidates.empty and BOOK_RATING_COUNT_COL in books.columns:
    try:
        candidates = books[(books[BOOK_RATING_COUNT_COL].astype(int) == 1234)]
    except Exception:
        pass

# additionally filter average_rating == 5.0
if BOOK_AVG_RATING_COL in books.columns:
    candidates = candidates[candidates[BOOK_AVG_RATING_COL] == 5.0]

print(f"Found {len(candidates)} candidate book(s) with rating_count=1234 and avg_rating=5.0.")

# If the dataset uses different names or numbers, we keep candidates empty-check
if candidates.empty:
    print("No exact candidate found with the exact filter. We'll broaden search to find reviews that contain the student hash.")
    # We'll search reviews for the hash directly (next step)
else:
    print("Candidate titles:")
    print(candidates[BOOK_TITLE_COL].fillna("").tolist())

# ---------- Step 4: scan reviews for the student hash (case-insensitive) ----------
# search for either lowercase or uppercase 8-hex substring
hash_pattern = re.compile(re.escape(student_hash8), flags=re.IGNORECASE)

matches = reviews[reviews[REV_TEXT_COL].fillna("").str.contains(student_hash8, case=False, na=False)]
print(f"Found {len(matches)} review(s) containing the hash {student_hash8} (case-insensitive).")

if matches.empty:
    print("No review contained the exact 8-char hash. You might need to check whether the dataset uses the lower-case/uppercase or full hash; searching for full hash prefix...")
    # try searching for first 8 of the full lowercase
    alt = student_hash_full_hex[:8]
    matches = reviews[reviews[REV_TEXT_COL].fillna("").str.contains(alt, case=False, na=False)]
    print("Found matches with alt:", len(matches))

# If matches found, get the book id(s)
if not matches.empty:
    book_ids_with_hash = matches[REV_BOOK_ID_COL].unique().tolist()
    print("Book IDs that have a review containing the student hash:", book_ids_with_hash)
    # fetch the book title(s)
    flagged_books = books[books[BOOK_ID_COL].isin(book_ids_with_hash)]
    print("Flagged book titles:")
    print(flagged_books[BOOK_TITLE_COL].tolist())
else:
    print("No matches found in reviews for the student hash. Stopping here — confirm dataset or check for different string variants.")

# ---------- Step 5: compute FLAG1 ----------
# We need the book title of the identified book. If multiple flagged_books exist, we'll operate on the first one.
if not matches.empty and not flagged_books.empty:
    book_title = str(flagged_books.iloc[0][BOOK_TITLE_COL])
    print("Identified book title:", book_title)
    first8 = first_nonspace_chars(book_title, n=8)
    flag1_full = sha256_hex(first8)
    print("First 8 non-space chars of title:", first8)
    print("FLAG1 (SHA256 of that string):", flag1_full)
else:
    book_title = None
    flag1_full = None

# ---------- Step 6: FLAG2 ----------
# FLAG2 is simply the hash found in the fake review (the 8 hex characters).
if not matches.empty:
    # pick the exact hash token from the first matching review (case-preserved)
    first_match_text = matches.iloc[0][REV_TEXT_COL]
    # find the token matching 8 hex
    found = re.search(r"[A-Fa-f0-9]{8}", first_match_text)
    if found:
        extracted_hash = found.group(0).upper()
    else:
        extracted_hash = student_hash8
    print("FLAG2 (hash found in review):", extracted_hash)
else:
    extracted_hash = None

# ---------- Step 7: Train a model to separate suspicious vs genuine reviews for the identified book ----------
# Heuristic labels for training:
# - Suspicious: rating == 5 AND short (len < 60 chars) AND uses superlatives (perfect, best, amazing, etc.)
# - Genuine: rating == 5 AND long (len > 80) AND contains domain words (plot, character, chapter, writing, style, theme)
# We will train on *all* reviews (across dataset) to get a model, then apply to reviews of the target book.

def contains_superlative(s):
    s = (s or "").lower()
    superlatives = ["best", "perfect", "amazing", "incredible", "fantastic", "awesome", "unbelievable", "brilliant", "excellent", "flawless"]
    return any(w in s for w in superlatives)

def contains_domain_word(s):
    s = (s or "").lower()
    domain_words = ["plot", "character", "characters", "plotline", "chapter", "writing", "style", "theme", "dialogue", "prose"]
    return any(w in s for w in domain_words)

reviews['text_len'] = reviews[REV_TEXT_COL].fillna("").apply(len)
reviews['has_super'] = reviews[REV_TEXT_COL].fillna("").apply(contains_superlative)
reviews['has_domain'] = reviews[REV_TEXT_COL].fillna("").apply(contains_domain_word)
reviews['rating_num'] = pd.to_numeric(reviews.get(REV_RATING_COL), errors='coerce')

# Build training set heuristically
suspicious_mask = (reviews['rating_num'] == 5) & (reviews['text_len'] < 60) & (reviews['has_super'])
genuine_mask = (reviews['rating_num'] == 5) & (reviews['text_len'] > 100) & (reviews['has_domain'])

train_df = reviews[suspicious_mask | genuine_mask].copy()
train_df['label'] = 0
train_df.loc[suspicious_mask, 'label'] = 1  # 1 => suspicious, 0 => genuine

print("Training examples:", len(train_df), "suspicious:", train_df['label'].sum(), "genuine:", (train_df['label']==0).sum())
if len(train_df) < 50:
    print("Warning: few heuristic training examples found. Consider relaxing heuristics or manually labelling more training data.")

# Train classifier (TF-IDF + logistic)
tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,2), token_pattern=r"(?u)\b\w+\b")
X = tfidf.fit_transform(train_df[REV_TEXT_COL].fillna("").values)
y = train_df['label'].values
clf = LogisticRegression(max_iter=200, solver='liblinear')
clf.fit(X, y)
print("Trained logistic regression on heuristic data.")

# ---------- Apply to reviews of the flagged book ----------
if book_title is not None:
    book_reviews = reviews[reviews[REV_BOOK_ID_COL].isin(book_ids_with_hash)].copy()
    book_reviews['pred_prob_suspicious'] = clf.predict_proba(tfidf.transform(book_reviews[REV_TEXT_COL].fillna("").values))[:,1]
    # Genuine = low predicted suspicion
    genuine_reviews = book_reviews[book_reviews['pred_prob_suspicious'] <= 0.3]   # threshold; adjust if needed
    print(f"Book reviews: {len(book_reviews)}; genuine (pred_prob<=0.3): {len(genuine_reviews)}")
else:
    genuine_reviews = pd.DataFrame()

# ---------- Step 8: SHAP analysis on genuine reviews ----------
# Use LinearExplainer (works well for linear models)
if not genuine_reviews.empty:
    X_background = X[:min(500, X.shape[0])]  # background from training set
    explainer = shap.LinearExplainer(clf, X_background, feature_dependence="independent")
    X_genuine = tfidf.transform(genuine_reviews[REV_TEXT_COL].fillna("").values)
    shap_vals = explainer.shap_values(X_genuine)  # shape: (n_samples, n_features) for binary
    # For logistic regression binary, shap_vals is an array shape (n_samples, n_features)
    # We want features where mean SHAP is negative (reduce suspicion)
    mean_shap = np.array(shap_vals).mean(axis=0)  # one value per feature
    # get feature names
    try:
        feature_names = tfidf.get_feature_names_out()
    except:
        feature_names = np.array([f"f{i}" for i in range(mean_shap.shape[0])])
    # find top 30 features with most negative mean shap (reduce suspicion)
    idx_sorted = np.argsort(mean_shap)  # ascending: most negative first
    top_reduce = idx_sorted[:30]
    top_words_reduce = [(feature_names[i], mean_shap[i]) for i in top_reduce[:10]]
    print("Top features that REDUCE suspicion (word, mean_SHAP):")
    for w, v in top_words_reduce[:10]:
        print(w, v)
    # pick top 3 words
    top3_words = [feature_names[i] for i in top_reduce[:3]]
    print("Selected top-3 words that reduce suspicion:", top3_words)
else:
    top3_words = []

# ---------- Step 9: make FLAG3 (you must insert your numeric ID) ----------
# Concatenate (words without spaces) + your numeric ID, then take SHA256 and first 10 hex chars
your_numeric_id = "<YOUR_NUMERIC_ID>"   # <-- replace this with your numeric ID, e.g., "001" or "160"
concatenated = "".join(top3_words) + str(your_numeric_id)
flag3_full = sha256_hex(concatenated)
flag3_prefix10 = flag3_full[:10]
print("Concatenation for FLAG3:", concatenated)
print("FLAG3 (first 10 hex of SHA256):", flag3_prefix10)

# ---------- Step 10: Save flags to flags.txt (example format) ----------
with open("flags.txt", "w") as f:
    f.write("FLAG1 = " + (flag1_full or "NOT_FOUND") + "\n")
    f.write("FLAG2 = FLAG2{" + (extracted_hash or "NOT_FOUND") + "}\n")
    f.write("FLAG3 = FLAG3{" + flag3_prefix10 + "}\n")

print("flags.txt written. Inspect the file for the final flags (remember to replace placeholder numeric ID in the code before generating final FLAG3).")
