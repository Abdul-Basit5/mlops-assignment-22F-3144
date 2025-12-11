# src/train.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import os

os.makedirs("models", exist_ok=True)

# Load dataset
df = pd.read_csv("data/dataset.csv")

# Normalize possible header name
if "Species" in df.columns and "species" not in df.columns:
    df.rename(columns={"Species": "species"}, inplace=True)

# Encode labels (just in case)
le = LabelEncoder()
df["species"] = le.fit_transform(df["species"].astype(str))

X = df.drop("species", axis=1)
y = df["species"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train a simple model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Save model and a metrics file
joblib.dump(model, "models/model.pkl")

acc = model.score(X_test, y_test)
with open("models/metrics.txt", "w") as f:
    f.write(f"accuracy: {acc:.4f}\n")

print(f"Model saved to models/model.pkl, test accuracy: {acc:.4f}")
