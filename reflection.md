# Reflection

This challenge involved a combination of data filtering, text mining, machine learning, and interpretability, all within a forensic context. The first task—identifying the manipulated book—required computing a custom hash and locating a hidden marker in the reviews dataset. This step highlighted how hidden signals can be embedded in large text corpora and how metadata such as rating distributions can reveal anomalies.

The second step focused on extracting the single fake review containing the injected hash value. Although simple, this illustrated an important real-world idea: malicious actors often hide signals or fingerprints inside otherwise normal-looking data.

The third step was the most technically interesting. By training a classifier to differentiate suspicious versus genuine reviews, I simulated how automated systems can detect manipulated or artificially inflated content. The use of TF-IDF and Logistic Regression provided a clear, explainable decision boundary. SHAP analysis further allowed me to interpret which words *reduced* suspicion, providing transparency into model decisions. The requirement to use only “genuine” reviews emphasized responsible use of explainability tools.

Overall, this challenge demonstrated how AI can be used both to uncover manipulation and to justify decisions with interpretability. It combined classical NLP techniques with modern interpretability frameworks in a meaningful way.
