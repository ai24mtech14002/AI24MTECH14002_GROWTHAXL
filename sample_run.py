import hashlib
import pandas as pd
import numpy as np
import re
import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import shap
import nltk
nltk.download('punkt', quiet=True)

# ---------- Config (user-provided schema) ----------
BOOKS_CSV = "books.csv"
REVIEWS_CSV = "reviews.csv"

# book columns (from your message)
BOOK_TITLE_COL = "title"
# identificaton column: use parent_asin in books
BOOK_ID_COL = "parent_asin"
BOOK_AVG_RATING_COL = "average_rating"
BOOK_RATING_COUNT_COL = "rating_number"

# review columns (from your message)
REV_TEXT_COL = "text"
REV_RATING_COL = "rating"
# reviews include both 'asin' and 'parent_asin' — prefer parent_asin, fall back to asin
REV_BOOK_ID_PREFERRED = "parent_asin"
REV_BOOK_ID_FALLBACK = "asin"

# ---------- Student string (your input) ----------
student_string = "STU039"   # <- you've told me this is your hash string
# compute SHA256 and 8-char token
def sha256_hex(s: str):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

student_hash_full_hex = sha256_hex(student_string)
student_hash8 = student_hash_full_hex[:8].upper()

print("Student string:", student_string)
print("SHA256(full) (hex):", student_hash_full_hex)
print("Student HASH (first 8 hex, UPPER):", student_hash8)
print()

# ---------- Robust CSV loading helper ----------
def try_read_csv(path):
    # try comma, then tab, then infer
    for sep in [",", "\t", None]:
        try:
            if sep is None:
                df = pd.read_csv(path, sep=None, engine="python", dtype=str)
            else:
                df = pd.read_csv(path, sep=sep, dtype=str)
            print(f"Loaded {path} with sep={sep!r}, shape={df.shape}")
            return df
        except Exception as e:
            # continue to next separator
            last_e = e
    raise last_e

if not os.path.exists(BOOKS_CSV) or not os.path.exists(REVIEWS_CSV):
    raise FileNotFoundError(f"Make sure {BOOKS_CSV} and {REVIEWS_CSV} are present in this folder.")

books = try_read_csv(BOOKS_CSV)
reviews = try_read_csv(REVIEWS_CSV)

# show columns we expect vs what's present
print("\nBooks columns available:", list(books.columns))
print("Reviews columns available:", list(reviews.columns))
print()

# ---------- Normalise and select the review -> book id column ----------
if REV_BOOK_ID_PREFERRED in reviews.columns:
    REV_BOOK_ID_COL = REV_BOOK_ID_PREFERRED
elif REV_BOOK_ID_FALLBACK in reviews.columns:
    REV_BOOK_ID_COL = REV_BOOK_ID_FALLBACK
else:
    # if neither present, try to find a likely column name
    found = None
    for trycol in ["parent_asin", "asin", "book_id", "bookId"]:
        if trycol in reviews.columns:
            found = trycol
            break
    if found is None:
        raise KeyError("Couldn't find an ASIN/parent ASIN column in reviews. Please check reviews.csv columns.")
    REV_BOOK_ID_COL = found

# similarly ensure BOOK_ID_COL exists in books (fall back to 'asin' if needed)
if BOOK_ID_COL not in books.columns:
    if "asin" in books.columns:
        BOOK_ID_COL = "asin"
    elif "parent_asin" in books.columns:
        BOOK_ID_COL = "parent_asin"
    else:
        # pick first column that looks like ASIN
        possible = [c for c in books.columns if "asin" in c.lower() or "parent" in c.lower()]
        if possible:
            BOOK_ID_COL = possible[0]
        else:
            raise KeyError("Couldn't find ASIN/parent ASIN column in books. Please check books.csv columns.")

print("Using BOOK_ID_COL (books):", BOOK_ID_COL)
print("Using REV_BOOK_ID_COL (reviews):", REV_BOOK_ID_COL)
print()

# ---------- Ensure numeric rating columns are numeric where applicable ----------
if BOOK_RATING_COUNT_COL in books.columns:
    try:
        books[BOOK_RATING_COUNT_COL] = pd.to_numeric(books[BOOK_RATING_COUNT_COL], errors="coerce")
    except Exception:
        pass

if BOOK_AVG_RATING_COL in books.columns:
    try:
        books[BOOK_AVG_RATING_COL] = pd.to_numeric(books[BOOK_AVG_RATING_COL], errors="coerce")
    except Exception:
        pass

if REV_RATING_COL in reviews.columns:
    try:
        reviews[REV_RATING_COL] = pd.to_numeric(reviews[REV_RATING_COL], errors="coerce")
    except Exception:
        pass

# ---------- Step: find reviews that contain the student hash ----------
hash_token = student_hash8
matches = reviews[reviews[REV_TEXT_COL].fillna("").str.contains(hash_token, case=False, na=False)]
print(f"Found {len(matches)} review(s) containing the hash {hash_token} (case-insensitive).")

# if none found, also try searching for lowercase prefix of full hex (defensive)
if matches.empty:
    alt = student_hash_full_hex[:8]
    matches = reviews[reviews[REV_TEXT_COL].fillna("").str.contains(alt, case=False, na=False)]
    print(f"After alternate search found {len(matches)} matches with token {alt}.")

if matches.empty:
    print("No review containing the hash was found. Please confirm the dataset contains the manipulated review.")
else:
    # show a preview of the first few matching reviews
    print("\n--- Example matched review(s) ---")
    for i, row in matches.head(5).iterrows():
        print(f"index={i}, {REV_BOOK_ID_COL}={row.get(REV_BOOK_ID_COL)}, rating={row.get(REV_RATING_COL)}")
        txt = str(row.get(REV_TEXT_COL))[:400].replace("\n"," ")
        print("text preview:", txt)
        print("-" * 60)

# ---------- Collect book(s) that have those matched reviews ----------
book_ids_with_hash = matches[REV_BOOK_ID_COL].dropna().unique().tolist() if not matches.empty else []
print("\nBook IDs with matching reviews:", book_ids_with_hash)

flagged_books = books[books[BOOK_ID_COL].isin(book_ids_with_hash)] if book_ids_with_hash else pd.DataFrame()
print("Number of flagged books found in books.csv:", len(flagged_books))
if not flagged_books.empty:
    print("Flagged book titles (first 5):")
    print(flagged_books[BOOK_TITLE_COL].fillna("").head(5).tolist())

# ---------- FLAG1: compute SHA256 of first 8 non-space chars of the identified book's title ----------
def first_nonspace_chars(s: str, n=8):
    return "".join(str(s).split())[:n]

if not flagged_books.empty:
    book_title = str(flagged_books.iloc[0].get(BOOK_TITLE_COL, ""))
    first8 = first_nonspace_chars(book_title, n=8)
    flag1_full = sha256_hex(first8)
    print("\nIdentified book title:", book_title)
    print("First 8 non-space chars:", first8)
    print("FLAG1 (SHA256 of first8):", flag1_full)
else:
    book_title = None
    flag1_full = None
    print("\nFLAG1: NOT FOUND (no flagged book).")

# ---------- FLAG2: the 8-hex token found in the fake review ----------
if not matches.empty:
    # extract first 8-hex substring from first matching review text
    first_match_text = matches.iloc[0][REV_TEXT_COL] or ""
    found = re.search(r"[A-Fa-f0-9]{8}", first_match_text)
    if found:
        extracted_hash = found.group(0).upper()
    else:
        extracted_hash = hash_token.upper()
    print("FLAG2 (hash from review):", extracted_hash)
else:
    extracted_hash = None

# ---------- Machine learning heuristic labels and model ----------
# Create features used by heuristics
reviews['text_len'] = reviews[REV_TEXT_COL].fillna("").apply(len)
def contains_superlative(s):
    s = (s or "").lower()
    superlatives = ["best", "perfect", "amazing", "incredible", "fantastic", "awesome", "unbelievable", "brilliant", "excellent", "flawless", "love"]
    return any(w in s for w in superlatives)

def contains_domain_word(s):
    s = (s or "").lower()
    domain_words = ["plot", "character", "characters", "plotline", "chapter", "writing", "style", "theme", "dialogue", "prose", "story"]
    return any(w in s for w in domain_words)

reviews['has_super'] = reviews[REV_TEXT_COL].fillna("").apply(contains_superlative)
reviews['has_domain'] = reviews[REV_TEXT_COL].fillna("").apply(contains_domain_word)
reviews['rating_num'] = pd.to_numeric(reviews.get(REV_RATING_COL), errors='coerce')

suspicious_mask = (reviews['rating_num'] == 5) & (reviews['text_len'] < 60) & (reviews['has_super'])
genuine_mask = (reviews['rating_num'] == 5) & (reviews['text_len'] > 100) & (reviews['has_domain'])

train_df = reviews[suspicious_mask | genuine_mask].copy()
if train_df.empty:
    print("\nWarning: heuristic training set empty. Try relaxing heuristics or provide manual labels.")
else:
    train_df['label'] = 0
    train_df.loc[suspicious_mask, 'label'] = 1  # 1 => suspicious, 0 => genuine
    print(f"\nHeuristic training examples: total={len(train_df)}, suspicious={train_df['label'].sum()}, genuine={(train_df['label']==0).sum()}")

    # TF-IDF + Logistic Regression
    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,2), token_pattern=r"(?u)\b\w+\b")
    X = tfidf.fit_transform(train_df[REV_TEXT_COL].fillna("").values)
    y = train_df['label'].values
    clf = LogisticRegression(max_iter=300, solver='liblinear')
    clf.fit(X, y)
    print("Trained logistic regression classifier on heuristic labels.")

    # Apply to reviews of the flagged book(s)
    if book_ids_with_hash:
        book_reviews = reviews[reviews[REV_BOOK_ID_COL].isin(book_ids_with_hash)].copy()
        if not book_reviews.empty:
            book_reviews['pred_prob_suspicious'] = clf.predict_proba(tfidf.transform(book_reviews[REV_TEXT_COL].fillna("").values))[:,1]
            # Genuine threshold (tuneable)
            genuine_reviews = book_reviews[book_reviews['pred_prob_suspicious'] <= 0.3]
            print(f"Book reviews total={len(book_reviews)}, predicted genuine (prob<=0.3)={len(genuine_reviews)}")
        else:
            genuine_reviews = pd.DataFrame()
    else:
        genuine_reviews = pd.DataFrame()

    # SHAP on genuine reviews (if present)
    top3_words = []
    if not genuine_reviews.empty:
        # Use small background sample (from training set)
        bg_samples = min(500, X.shape[0])
        X_background = X[:bg_samples]
        explainer = shap.LinearExplainer(clf, X_background, feature_dependence="independent")
        X_genuine = tfidf.transform(genuine_reviews[REV_TEXT_COL].fillna("").values)
        shap_vals = explainer.shap_values(X_genuine)  # returns (n_samples, n_features)
        mean_shap = np.array(shap_vals).mean(axis=0)
        try:
            feature_names = tfidf.get_feature_names_out()
        except:
            feature_names = np.array([f"f{i}" for i in range(mean_shap.shape[0])])
        idx_sorted = np.argsort(mean_shap)  # ascending: most negative first
        top_reduce = idx_sorted[:30]
        top3_words = [feature_names[i] for i in top_reduce[:3]]
        print("\nTop features (words/phrases) that REDUCE suspicion (top 10 shown):")
        for i in top_reduce[:10]:
            print(feature_names[i], mean_shap[i])
        print("\nTop-3 words selected for FLAG3:", top3_words)
    else:
        print("\nNo genuine reviews found for SHAP analysis; FLAG3 cannot be generated automatically.")

# ---------- FLAG3: build from top3 words + numeric ID ----------
your_numeric_id = "039"   
if top3_words and your_numeric_id and your_numeric_id != "<YOUR_NUMERIC_ID>":
    concat = "".join([w.replace(" ", "") for w in top3_words]) + str(your_numeric_id)
    flag3_full = sha256_hex(concat)
    flag3_prefix10 = flag3_full[:10]
    print("\nConcatenation for FLAG3:", concat)
    print("FLAG3 (first 10 hex of SHA256):", flag3_prefix10)
else:
    concat = None
    flag3_full = None
    flag3_prefix10 = None
    if not top3_words:
        print("\nFLAG3: cannot compute because top3 words not available.")
    else:
        print("\nFLAG3: numeric ID placeholder not replaced. Set your_numeric_id variable in the script and re-run to compute FLAG3.")

# ---------- Save flags (best-effort) ----------
with open("flags.txt", "w") as f:
    f.write("FLAG1 = " + (flag1_full or "NOT_FOUND") + "\n")
    f.write("FLAG2 = FLAG2{" + (extracted_hash or "NOT_FOUND") + "}\n")
    if flag3_prefix10:
        f.write("FLAG3 = FLAG3{" + flag3_prefix10 + "}\n")
    else:
        f.write("FLAG3 = NOT_COMPUTED (set numeric id and ensure SHAP finished)\n")

print("\nWrote flags.txt. Inspect it. If FLAG3 is NOT_COMPUTED, replace your_numeric_id in this script and re-run.")
