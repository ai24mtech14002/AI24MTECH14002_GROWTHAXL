## Files
 - `solver.py` — main script (this file)
 - `books.csv` — book metadata (place in same folder)
 - `reviews.csv` — review dataset (place in same folder)
 - `flags.txt` — output created by `solver.py` after running
 - `requirements.txt` (suggested packages, see below)

## What this does (short)
1. Computes student hash from the string `STU039` (SHA256, uses first 8 hex chars).
2. Scans `reviews.csv` for any review that contains that 8-hex token.
3. Identifies the book(s) (by parent ASIN / ASIN) that have those reviews.
4. Computes `FLAG1`: SHA256 of the first 8 non-space characters of that book's title.
5. Extracts `FLAG2`: the 8-hex token found in the fake review.
6. Builds a small heuristic training set (suspicious vs genuine 5-star reviews), trains TF-IDF + LogisticRegression, applies it to the target book's reviews, uses SHAP on the predicted genuine reviews to find top-3 words that reduce suspicion, and builds `FLAG3` from those words + your numeric ID (you must set your numeric ID in `solver.py`).

## How to run
1. Ensure Python 3.9+ is available.
2. (Recommended) create & activate a virtual environment:
   ```bash
   python -m venv venv     
   venv\Scripts\activate         
