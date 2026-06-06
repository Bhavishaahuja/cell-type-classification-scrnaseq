import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# 1. Load Data

train = pd.read_csv(r"C:\Users\Bhavi\Downloads\CS 412 Final Project\cells_batch1_20000.csv")
test = pd.read_csv(r"C:\Users\Bhavi\Downloads\CS 412 Final Project\cells_batch2_10000.csv")

# 2. Feature Selection

feature_cols = [c for c in train.columns if c not in ["cell_id", "cell_type"]]

X = train[feature_cols]
y = train["cell_type"]
X_test = test[feature_cols]

X = X.fillna(0)
X_test = X_test.fillna(0)

# 3. Train/Validation Split

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# 4. Scaling (for LR & PCA pipeline)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
# 5. Baseline Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    n_jobs=-1,
    random_state=42,
    class_weight="balanced"
)

rf.fit(X_train, y_train)
y_val_pred = rf.predict(X_val)

baseline_rf_accuracy = accuracy_score(y_val, y_val_pred)
print("Baseline RF Accuracy:", baseline_rf_accuracy)
print(classification_report(y_val, y_val_pred))

# 6. Hyperparameter Tuning

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 20, 40],
    "min_samples_split": [2, 5],
}

rf_base = RandomForestClassifier(
    n_jobs=-1,
    random_state=42,
    class_weight="balanced"
)

grid = GridSearchCV(
    rf_base,
    param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X_train, y_train)
best_rf = grid.best_estimator_

y_val_pred_best = best_rf.predict(X_val)
tuned_rf_accuracy = accuracy_score(y_val, y_val_pred_best)

print("Best Hyperparameters:", grid.best_params_)
print("Tuned RF Accuracy:", tuned_rf_accuracy)
print(classification_report(y_val, y_val_pred_best))

# 7. Logistic Regression (Comparison Model)

logreg_pipe = Pipeline([
    ("scaler", StandardScaler(with_mean=False)),
    ("clf", LogisticRegression(max_iter=1000, multi_class="multinomial"))
])

logreg_pipe.fit(X_train, y_train)
y_val_pred_lr = logreg_pipe.predict(X_val)

logreg_accuracy = accuracy_score(y_val, y_val_pred_lr)
print("Logistic Regression Accuracy:", logreg_accuracy)
print(classification_report(y_val, y_val_pred_lr))
# 8. Train Final Model on Full Training Data

best_model = best_rf

best_model.fit(X, y)

test_pred = best_model.predict(X_test)

# 9. Save Predictions for Submission

predictions = pd.DataFrame({
    "cell_id": test["cell_id"],
    "predicted_cell_type": test_pred
})

predictions.to_csv("Ahuja_Bhavisha_653160655.csv", index=False)

