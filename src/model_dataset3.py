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

TRAIN_X = "data/TrainData3.txt"
TRAIN_Y = "data/TrainLabel3.txt"
TEST_X  = "data/TestData3.txt"

x_train, y_train, x_test = load_dataset(TRAIN_X, TRAIN_Y, TEST_X)

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np

pipeline3_rf = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        random_state=42
    )),
])

cv3 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores3_rf = cross_val_score(
    pipeline3_rf, x_train, y_train,
    cv=cv3, scoring="accuracy", n_jobs=-1
)

# Fit final RF model on all training data
pipeline3_rf.fit(x_train, y_train)

# Predict on test set
test_pred3 = pipeline3_rf.predict(x_test).astype(int)

# Save to file
np.savetxt("results_dataset3.txt", test_pred3, fmt="%d")

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
    pipeline3_rf, x_train, y_train,
    cv=cv3, scoring="accuracy", n_jobs=-1
)
print(f"Cross-Validated Accuracy: {cv_acc.mean():.3f} ± {cv_acc.std():.3f}")

# 2️⃣ Macro-F1 (Cross-Validation)
cv_f1 = cross_val_score(
    pipeline3_rf, x_train, y_train,
    cv=cv3, scoring="f1_macro", n_jobs=-1
)
print(f"Macro-F1 Score: {cv_f1.mean():.3f} ± {cv_f1.std():.3f}")

# 3️⃣ Balanced Accuracy (Cross-Validation)
cv_bal_acc = cross_val_score(
    pipeline3_rf, x_train, y_train,
    cv=cv3, scoring="balanced_accuracy", n_jobs=-1
)
print(f"Balanced Accuracy: {cv_bal_acc.mean():.3f} ± {cv_bal_acc.std():.3f}")

# 4️⃣ Cross-validated Predictions for Detailed Metrics
y_pred_cv = cross_val_predict(
    pipeline3_rf,
    x_train, y_train,
    cv=cv3, n_jobs=-1
)

# 5️⃣ Per-class Metrics
print("\nClassification Report:")
print(classification_report(y_train, y_pred_cv, digits=3))

# 6️⃣ Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_train, y_pred_cv))

print("====================================================\n")