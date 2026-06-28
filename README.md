# 🧠 Fake News Detection Web App

An AI-powered web application that detects whether a news article is **real or fake** using Machine Learning and Flask.

---

## 🚀 Live Demo
(Add your deployed link here after Render deployment)

https://your-app.onrender.com

---

## 📌 Features

- 📰 Fake vs Real News Classification
- 📊 Confidence Score Output
- 🧠 Explainable AI (Top influencing words)
- 🌙 Dark Mode / Light Mode Toggle
- ⚡ Fast real-time prediction
- 🌐 Flask web application

---

## 🛠 Tech Stack

- Python 🐍
- Flask 🌐
- Scikit-learn 🤖
- NumPy
- HTML, CSS, JavaScript
- Bootstrap 5

---

## 📂 Project Structure
```
fake-news-detector/
│
├── app.py
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
├── wsgi.py
│
├── templates/
│ └── index.html
│
├── static/
│ └── style.css


---

## ⚙️ How It Works

1. User enters news text
2. Text is converted using TF-IDF vectorizer
3. ML model predicts Fake or Real
4. System shows:
   - Prediction result
   - Confidence score
   - Important words influencing decision

---

## 🚀 Run Locally

### 1. Clone repository
```bash
git clone https://github.com/GandeRani/fake-news-detector.git

2. Install dependencies
pip install -r requirements.txt
3. Run app
python app.py
4. Open in browser
http://127.0.0.1:5000
📊 Model Details
Algorithm: Logistic Regression
Feature Extraction: TF-IDF
Dataset: Fake & Real News Dataset (Kaggle)
Accuracy: ~98%
🧠 Future Improvements
Deploy on Render / Railway
Add BERT-based model
Add URL-based news detection
Improve explanation visuals
👨‍💻 Author

Rani Gande
AI / ML Developer
