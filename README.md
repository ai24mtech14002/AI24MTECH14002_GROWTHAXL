# CTF_STU039 — Capture The Flag Solution

This repository contains my solution for the AI Forensics "Capture the Flag" challenge.

## Overview
The goal of the challenge was to identify a manipulated book in a dataset, detect a hidden review containing a custom hash, and then perform SHAP-based authenticity analysis on associated reviews to derive three separate flags.

The challenge was completed in three main stages:

---

## Step 1 — Identify the Manipulated Book (FLAG1)
- First, I computed the SHA256 hash of **"STU039"**, taking the first 8 uppercase hex characters:
DBCC0D53
- Then I filtered books with:
- `rating_number = 1234`
- `average_rating = 5.0`
- I scanned all their reviews for this hash and identified the fake review.
- I extracted the first 8 non-space characters from the book title:
- `"Every Time a Bell Rings"` → `"EveryTim"`
- I computed `SHA256("EveryTim")` → **FLAG1**.

---

## Step 2 — Find the Fake Review (FLAG2)
- The review containing my hash `"DBCC0D53"` was considered the injected fake review.
- FLAG2 is simply:
FLAG2{DBCC0D53}

---

## Step 3 — SHAP-Based Authenticity Analysis (FLAG3)
- I trained a text classification model (TF-IDF + Logistic Regression) to differentiate:
- *Suspicious reviews*: very short, overly positive, or containing superlatives  
- *Genuine reviews*: longer, descriptive, domain-specific language  
- I applied SHAP only on the **genuine reviews** to identify words that *reduce* suspicion.
- The top three authenticity-inducing words were:
bell, belle, best
- I concatenated these with my numeric ID `"039"` → `"bellbellebest039"`
- Taking SHA256 and extracting the first 10 hex characters gave **FLAG3**.

---

## Final Flags
FLAG1 = a1f5042c13b6621671e120065aa91e2bf884fc1b1c3973b49974a60e5562d8fb
FLAG2 = FLAG2{DBCC0D53}
FLAG3 = FLAG3{9ba28c2941}


---


