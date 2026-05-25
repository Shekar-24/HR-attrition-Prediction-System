import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Config 
DATA_PATH  = "HR-Employee-Attrition.csv"
MODEL_PATH = "attrition_model.pkl"
FEATURES   = ["Age", "MonthlyIncome", "TotalWorkingYears", "YearsAtCompany"]
TARGET     = "Attrition"

# Load 
df = pd.read_csv(DATA_PATH)
X  = df[FEATURES]
y  = (df[TARGET] == "Yes").astype(int)          # 1 = Yes, 0 = No

# Split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Train 
model = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.08,
    max_depth=4,
    subsample=0.85,
    random_state=42,
)
model.fit(X_train, y_train)

# Evaluate 
y_pred = model.predict(X_test)
print("=" * 50)
print("  Gradient Boosting — Attrition Model")
print("=" * 50)
print(f"  Accuracy : {accuracy_score(y_test, y_pred):.2%}")
print()
print(classification_report(y_test, y_pred, target_names=["No", "Yes"]))

# Dump 
joblib.dump(model, MODEL_PATH)
print(f"✅  Model saved → {MODEL_PATH}")
