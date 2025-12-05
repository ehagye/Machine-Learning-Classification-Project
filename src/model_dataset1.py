# importing the data first
from pathlib import Path
import sys
import os
import numpy as np

root = Path.cwd()
while root != root.parent and not (root / "pyproject.toml").exists():
    root = root.parent

sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "src"))
os.chdir(root)

from data_loader import load_dataset

TRAIN_X = "data/TrainData1.txt"
TRAIN_Y = "data/TrainLabel1.txt"
TEST_X  = "data/TestData1.txt"

x_train, y_train, x_test = load_dataset(TRAIN_X, TRAIN_Y, TEST_X)

# creating the pipeline for preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=0.95)),  # keep 95% of variance
    ("clf", SVC(kernel="linear", class_weight="balanced", random_state=42)),
])

# using stratifiedKFold because of class imbalance.
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

scores = cross_val_score(
    pipeline,
    x_train,
    y_train,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
)

# finally training on all the training data and predicting test labels

# Fit final model on all training data
pipeline.fit(x_train, y_train)

# Predict on test set
test_pred = pipeline.predict(x_test)

# Make sure they are ints
test_pred = test_pred.astype(int)

from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)
import numpy as np

print("\n================ Evaluation Metrics ================")

# 1️⃣ Accuracy (Cross-Validation)
cv_acc = cross_val_score(
    pipeline, x_train, y_train,
    cv=cv, scoring="accuracy", n_jobs=-1
)
print(f"Cross-Validated Accuracy: {cv_acc.mean():.3f} ± {cv_acc.std():.3f}")

# 2️⃣ Macro-F1 (Cross-Validation)
cv_f1 = cross_val_score(
    pipeline, x_train, y_train,
    cv=cv, scoring="f1_macro", n_jobs=-1
)
print(f"Macro-F1 Score: {cv_f1.mean():.3f} ± {cv_f1.std():.3f}")

# 3️⃣ Balanced Accuracy (Cross-Validation)
cv_bal_acc = cross_val_score(
    pipeline, x_train, y_train,
    cv=cv, scoring="balanced_accuracy", n_jobs=-1
)
print(f"Balanced Accuracy: {cv_bal_acc.mean():.3f} ± {cv_bal_acc.std():.3f}")

# 4️⃣ Cross-validated Predictions for Detailed Metrics
y_pred_cv = cross_val_predict(
    pipeline,
    x_train, y_train,
    cv=cv, n_jobs=-1
)

# 5️⃣ Per-class Metrics
print("\nClassification Report:")
print(classification_report(y_train, y_pred_cv, digits=3))

# 6️⃣ Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_train, y_pred_cv))

print("====================================================\n")

