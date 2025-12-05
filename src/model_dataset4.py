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

TRAIN_X = "data/TrainData4.txt"
TRAIN_Y = "data/TrainLabel4.txt"
TEST_X  = "data/TestData4.txt"

x_train, y_train, x_test = load_dataset(TRAIN_X, TRAIN_Y, TEST_X)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np

pipeline4_RFC_tuned = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42, k_neighbors=3)),  # key add
    ("clf", RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=1,
        class_weight=None,  # no need—SMOTE balances classes
        random_state=42
    )),
])

pipeline4_SVM = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("clf", SVC(
        kernel="rbf",
        C=3,            # slightly stronger boundary
        gamma="scale",
        class_weight="balanced",
        random_state=42
    )),
])

cv4 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores4_RFC_tuned = cross_val_score(
    pipeline4_RFC_tuned,
    x_train, y_train,
    cv=cv4,
    scoring="accuracy",   # <-- NEW SCORING METRIC
    n_jobs=-1
)

scores4_SVM = cross_val_score(
    pipeline4_SVM,
    x_train, y_train,
    cv=cv4, scoring="accuracy", n_jobs=-1
)

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_predict

y_pred_cv = cross_val_predict(
    pipeline4_RFC_tuned,
    x_train, y_train,
    cv=cv4,
    n_jobs=-1
)

# ==== FINAL MODEL TRAINING AND PREDICTION FOR DATASET 4 ====

# Fit the model on the full training set
pipeline4_RFC_tuned.fit(x_train, y_train)

# Predict labels for the test set
test_pred_4 = pipeline4_RFC_tuned.predict(x_test)

# Convert predictions to int format
test_pred_4 = test_pred_4.astype(int)

# Save output file in required assignment format
output_path = "results_dataset4.txt"
np.savetxt(output_path, test_pred_4, fmt="%d")

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
    pipeline4_RFC_tuned, x_train, y_train,
    cv=cv4, scoring="accuracy", n_jobs=-1
)
print(f"Cross-Validated Accuracy: {cv_acc.mean():.3f} ± {cv_acc.std():.3f}")

# 2️⃣ Macro-F1 (Cross-Validation)
cv_f1 = cross_val_score(
    pipeline4_RFC_tuned, x_train, y_train,
    cv=cv4, scoring="f1_macro", n_jobs=-1
)
print(f"Macro-F1 Score: {cv_f1.mean():.3f} ± {cv_f1.std():.3f}")

# 3️⃣ Balanced Accuracy (Cross-Validation)
cv_bal_acc = cross_val_score(
    pipeline4_RFC_tuned, x_train, y_train,
    cv=cv4, scoring="balanced_accuracy", n_jobs=-1
)
print(f"Balanced Accuracy: {cv_bal_acc.mean():.3f} ± {cv_bal_acc.std():.3f}")

# 4️⃣ Cross-validated Predictions for Detailed Metrics
y_pred_cv = cross_val_predict(
    pipeline4_RFC_tuned,
    x_train, y_train,
    cv=cv4, n_jobs=-1
)

# 5️⃣ Per-class Metrics
print("\nClassification Report:")
print(classification_report(y_train, y_pred_cv, digits=3))

# 6️⃣ Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_train, y_pred_cv))

print("====================================================\n")