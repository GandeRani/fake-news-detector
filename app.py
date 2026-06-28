from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model + vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["news"]

    # Transform input
    vec = vectorizer.transform([text])

    # Prediction
    prediction = model.predict(vec)

    # Confidence
    prob = model.predict_proba(vec)[0]
    confidence = float(np.max(prob)) * 100

    # Explainability (top words)
    feature_names = vectorizer.get_feature_names_out()
    vec_array = vec.toarray()[0]

    word_weights = []

    for i, value in enumerate(vec_array):
        if value > 0:
            word_weights.append((feature_names[i], value))

    word_weights = sorted(word_weights, key=lambda x: x[1], reverse=True)

    top_words = [w[0] for w in word_weights[:8]]

    # Result
    if prediction[0] == 1:
        result = "REAL NEWS 🟢"
    else:
        result = "FAKE NEWS 🔴"

    return render_template(
        "index.html",
        prediction_text=result,
        confidence=round(confidence, 2),
        explain_words=top_words
    )

if __name__ == "__main__":
    app.run(debug=True)