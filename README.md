# 🔍 TruthLens — Fake News Detection System

> An AI-powered web application that detects fake news using NLP and Machine Learning, built with Python, Flask, Scikit-learn, and Bootstrap 5.

---

## 📌 Project Overview

TruthLens is an end-to-end fake news detection system that takes a news article as input and classifies it as **FAKE** or **REAL** using natural language processing and machine learning. The project covers the complete ML lifecycle — from data preprocessing and EDA to model training, evaluation, serialisation, and deployment via a Flask web application.

**Use Cases:** GitHub portfolio · College final year project · Hackathon submission · AI/ML internship interview demo

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧹 NLP Pipeline | Lowercase, URL removal, punctuation, stopwords, tokenisation, lemmatisation |
| 📊 EDA Dashboard | Distribution charts, word frequency, article length analysis, word clouds |
| 🤖 3 ML Models | Logistic Regression, Linear SVM, Multinomial Naive Bayes |
| 🏆 Auto Model Selection | Best model (by F1 score) automatically saved and deployed |
| 🌐 Flask Web App | Clean, responsive web interface with Bootstrap 5 |
| 📈 Analytics Dashboard | Chart.js powered live analytics with prediction stats |
| 🗂️ Prediction History | Every prediction logged to CSV with timestamp & confidence |
| 📦 GitHub Ready | Complete folder structure, requirements.txt, and documentation |

---

## 🛠️ Tech Stack

```
Backend    : Python 3.10+, Flask 3.0
ML / NLP   : Scikit-learn, NLTK, TF-IDF Vectoriser
EDA        : Matplotlib, Seaborn, WordCloud
Frontend   : Bootstrap 5, Chart.js, Bootstrap Icons
Serialise  : Joblib
Storage    : CSV (prediction history)
```

---

## 📁 Project Structure

```
Fake-News-Detection/
│
├── data/
│   ├── Fake.csv                  # Fake news dataset
│   └── True.csv                  # Real news dataset
│
├── models/                       # Auto-created after training
│   ├── model.pkl                 # Best serialised model
│   ├── vectorizer.pkl            # TF-IDF vectoriser
│   ├── metrics.csv               # All model metrics
│   └── best_model_name.txt       # Name of deployed model
│
├── static/
│   ├── images/                   # EDA charts & word clouds (auto-generated)
│   ├── css/
│   └── js/
│
├── templates/
│   ├── index.html                # Prediction page
│   └── dashboard.html            # Analytics dashboard
│
├── app.py                        # Flask web application
├── train.py                      # Full ML training pipeline
├── requirements.txt
├── README.md
└── prediction_history.csv        # Auto-created at runtime
```

---

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Fake-News-Detection.git
cd Fake-News-Detection
```

### 2. Create a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the Dataset

Place the dataset files inside the `data/` folder:

```
data/
├── Fake.csv
└── True.csv
```

> Download from: [Kaggle — Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

### 5. Train the Model

```bash
python train.py
```

This will:
- Preprocess the dataset (NLP pipeline)
- Generate EDA charts and word clouds → `static/images/`
- Train Logistic Regression, Linear SVM, and Multinomial Naive Bayes
- Print a model comparison table
- Save the best model → `models/model.pkl` and `models/vectorizer.pkl`

Expected output:
```
INFO | Loading datasets …
INFO | Running NLP preprocessing …
INFO | Running EDA …
INFO | Building TF-IDF features …
INFO | Training all models …

════════════════════════════════════════════════════
MODEL COMPARISON TABLE
════════════════════════════════════════════════════
                          accuracy  precision  recall     f1
Logistic Regression         98.76      98.81   98.74  98.77
Linear SVM                  99.12      99.10   99.15  99.12
Multinomial Naive Bayes     94.23      93.87   94.67  94.27

✅ Training complete!
   Best Model : Linear SVM
   Accuracy   : 99.12%
   F1 Score   : 99.12%
```

### 6. Run the Web Application

```bash
python app.py
```

Open your browser at: **http://localhost:5000**

---

## 📸 Screenshots

| Page | Description |
|---|---|
| Home / Detector | Paste any article and get an instant Fake/Real verdict |
| Result Panel | Shows verdict, confidence %, model name, and timestamp |
| Analytics Dashboard | KPI cards, Chart.js plots, EDA visualisations |
| Model Comparison Table | Accuracy, Precision, Recall, F1 for all 3 models |
| Prediction History | Every prediction stored with timestamp and confidence |

> *Screenshots will be added after the application is running.*

---

## 🔬 ML Pipeline — Phase by Phase

### Phase 1: NLP Preprocessing
Each raw article goes through:

| Step | Why |
|---|---|
| Lowercase | Normalises `News` == `news` |
| URL removal | URLs carry no semantic meaning |
| HTML stripping | Removes leftover markup |
| Punctuation removal | Punctuation adds noise to bag-of-words models |
| Special char removal | Keeps only alphabetic tokens |
| Number removal | Digits rarely signal fake vs real |
| Tokenisation | Splits text into individual words |
| Stopword removal | Filters high-frequency low-information words (`the`, `is`, …) |
| Lemmatisation | Reduces `running` → `run`, `articles` → `article` |

### Phase 3: Why TF-IDF over CountVectorizer?

- **CountVectorizer** gives raw frequencies — biased toward common words that appear everywhere.
- **TF-IDF** penalises words frequent across all documents (low IDF) while rewarding words distinctive to specific articles (high TF), making it far better at capturing the distinguishing vocabulary of fake vs real news.
- **Bigrams** (e.g. `"fake news"`, `"white house"`, `"breaking news"`) carry much richer context than individual words.

### Phase 4: Models Compared

| Model | Strength |
|---|---|
| Logistic Regression | Fast, interpretable, strong baseline |
| Linear SVM | Excellent for high-dimensional sparse text |
| Multinomial Naive Bayes | Very fast; good with TF-IDF proportions |

---

## 🌐 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Home page with prediction form |
| `/predict` | POST | Submit news text for classification |
| `/dashboard` | GET | Analytics dashboard |
| `/api/history` | GET | JSON — last N predictions |
| `/api/stats` | GET | JSON — aggregate stats + model metrics |

---

## 🔮 Future Enhancements

- [ ] Deep learning model (LSTM / BERT fine-tuning)
- [ ] Multi-language support
- [ ] Browser extension integration
- [ ] Real-time news feed scanning (NewsAPI)
- [ ] User authentication & personal history
- [ ] REST API with JWT tokens
- [ ] Dockerisation for one-command deployment
- [ ] Explainability (LIME / SHAP) for predictions

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first.

---

## 👤 Author

Built as an intermediate AI/ML project demonstrating end-to-end NLP + Flask development skills.

**Stack Highlights:** Python · Flask · Scikit-learn · NLTK · TF-IDF · Bootstrap 5 · Chart.js


## Dataset

The original Fake.csv and True.csv datasets are not included in this repository due to their large size.

Dataset Source:
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
