# fast_spam.py
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np


# 1) Load data
train1 = pd.read_csv("spam_train1.csv")   # v1=label, v2=text (+ extra unnamed cols)
train2 = pd.read_csv("spam_train2.csv")   # text + label (or label_num)
test   = pd.read_csv("spam_test.csv")     # message

# 2) Normalize training formats
def normalize_train(df):
    df = df.rename(columns={c: c.lower() for c in df.columns})
    if "text" in df.columns and ("label" in df.columns or "label_num" in df.columns):
        text = df["text"].fillna("")
        lab  = df["label"] if "label" in df.columns else df["label_num"]
    else:
        # train1 style (SMS Spam Collection): v1=ham/spam, v2=message text
        text = df["v2"].fillna("")
        lab  = df["v1"]
    y = lab.astype(str).str.strip().str.lower().map({"ham":0,"spam":1,"0":0,"1":1})
    return pd.DataFrame({"text": text, "label": y})

def normalize_test(df):
    df = df.rename(columns={c: c.lower() for c in df.columns})
    # your test file uses 'message' as the text column
    txt = df["text"] if "text" in df.columns else df["message"]
    return pd.DataFrame({"text": txt.fillna("")})

tr1 = normalize_train(train1)
tr2 = normalize_train(train2)
te  = normalize_test(test)

# Combine and deduplicate exact duplicates
train_df = pd.concat([tr1, tr2], ignore_index=True).drop_duplicates(subset=["text","label"])

# 3) Build model: HashingVectorizer (uni+bi-grams) + LinearSVC (fast & strong)
pipe = Pipeline([
    ("hash", HashingVectorizer(n_features=2**18, alternate_sign=False,
                               lowercase=True, analyzer="word", ngram_range=(1,2))),
    ("clf",  LinearSVC(class_weight="balanced", C=1.0, random_state=42)),
])

# 4) Train and predict
pipe.fit(train_df["text"], train_df["label"])
pred = pipe.predict(te["text"])

# 5) Write predictions file: One label per line as 'Ham' or 'Spam'
id2lab = {0: "Ham", 1: "Spam"}
with open("SeyumSpam.txt", "w", encoding="utf-8") as f:
    for v in pred:
        f.write(id2lab[int(v)] + "\n")
X = train_df["text"]
y = train_df["label"].astype(int)
Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

pipe.fit(Xtr, ytr)
# LinearSVC has no predict_proba; use decision_function → sigmoid for AUC
decision = pipe.decision_function(Xva)
proba = 1/(1+np.exp(-decision))
pred  = (proba >= 0.5).astype(int)

acc = accuracy_score(yva, pred)
prec, rec, f1, _ = precision_recall_fscore_support(yva, pred, average="binary", zero_division=0)
auc = roc_auc_score(yva, proba)

print({
  "val_accuracy": round(acc,4),
  "val_precision": round(prec,4),
  "val_recall": round(rec,4),
  "val_f1": round(f1,4),
  "val_roc_auc": round(auc,4),
})
print("Wrote SeyumSpam.txt")
