import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -----------------------------
# 1. Load dataset
# -----------------------------
true_df = pd.read_csv("True.csv")
fake_df = pd.read_csv("Fake.csv")

# -----------------------------
# 2. Add labels
# -----------------------------
true_df["label"] = 1   # real news
fake_df["label"] = 0   # fake news

# -----------------------------
# 3. Combine datasets
# -----------------------------
df = pd.concat([true_df, fake_df], axis=0)

# -----------------------------
# 4. Create text column
# -----------------------------
df["content"] = df["title"] + " " + df["text"]

df = df[["content", "label"]]
df = df.dropna()

print("Dataset loaded successfully")
print(df["label"].value_counts())

# -----------------------------
# 5. Split data
# -----------------------------
X = df["content"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 6. TF-IDF Vectorization
# -----------------------------
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -----------------------------
# 7. Train model
# -----------------------------
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# -----------------------------
# 8. Evaluate model
# -----------------------------
y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# -----------------------------
# 9. Save model + vectorizer
# -----------------------------
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model and vectorizer saved successfully")