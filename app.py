"""
app.py — Fake News Detection: Flask Web Application
=====================================================
Features:
  • /               → Home / Prediction page
  • /dashboard      → Analytics dashboard
  • /predict        → POST endpoint for news prediction
  • /api/history    → JSON endpoint for prediction history
  • /api/stats      → JSON endpoint for aggregate stats

Production practices used:
  • Blueprint-style modular routing
  • Centralised error handling (404, 500)
  • Input validation with clear user messages
  • Confidence score from decision function / predict_proba
  • Prediction history persisted to CSV
  • Model & metrics loaded once at startup (not per request)
"""

import os
import csv
import json
import logging
import re
import string
from datetime import datetime
from functools import lru_cache

import joblib
import nltk
import numpy as np
import pandas as pd
import requests
from flask import (Flask, jsonify, render_template,
                   request, redirect, url_for)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# NLTK setup
# ─────────────────────────────────────────────
_NLTK_PACKAGES = ["stopwords", "wordnet", "punkt", "omw-1.4", "punkt_tab"]
for _pkg in _NLTK_PACKAGES:
    nltk.download(_pkg, quiet=True)

STOP_WORDS  = set(stopwords.words("english"))
_lemmatizer = WordNetLemmatizer()

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
MODEL_PATH       = "models/model.pkl"
VECTORIZER_PATH  = "models/vectorizer.pkl"
METRICS_PATH     = "models/metrics.csv"
BEST_NAME_PATH   = "models/best_model_name.txt"
HISTORY_CSV      = "prediction_history.csv"
HISTORY_COLUMNS  = ["id", "timestamp", "snippet", "full_text",
                     "prediction", "confidence", "model_used"]
MAX_HISTORY      = 500   # keep only the last N predictions in memory
NEWS_API_KEY = "YOUR_NEWS_API_KEY"
# ─────────────────────────────────────────────
# Flask app
# ─────────────────────────────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET", "fnews-dev-secret-2024")
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024   # 2 MB

# ─────────────────────────────────────────────
# Load ML artefacts (once at startup)
# ─────────────────────────────────────────────
def _load_artefacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        log.warning("Model artefacts not found. Run train.py first.")
        return None, None, [], "N/A"

    model      = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    metrics = []
    if os.path.exists(METRICS_PATH):
        metrics = pd.read_csv(METRICS_PATH).to_dict(orient="records")

    best_model_name = "Unknown"
    if os.path.exists(BEST_NAME_PATH):
        with open(BEST_NAME_PATH) as f:
            best_model_name = f.read().strip()

    log.info(f"Loaded model: {best_model_name}")
    return model, vectorizer, metrics, best_model_name


MODEL, VECTORIZER, METRICS, BEST_MODEL_NAME = _load_artefacts()

# ─────────────────────────────────────────────
# Prediction history  (in-memory + CSV persistence)
# ─────────────────────────────────────────────
_prediction_history: list[dict] = []
_pred_id_counter = 0


def _init_history_csv():
    if not os.path.exists(HISTORY_CSV):
        with open(HISTORY_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=HISTORY_COLUMNS)
            writer.writeheader()


def _load_history_from_csv():
    global _pred_id_counter
    if not os.path.exists(HISTORY_CSV):
        return []
    df = pd.read_csv(HISTORY_CSV, encoding="utf-8")
    records = df.tail(MAX_HISTORY).to_dict(orient="records")
    if records:
        _pred_id_counter = max(r["id"] for r in records)
    return records


def _append_history(record: dict):
    with open(HISTORY_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_COLUMNS)
        writer.writerow(record)


_init_history_csv()
_prediction_history = _load_history_from_csv()

# ─────────────────────────────────────────────
# NLP preprocessing (mirrors train.py exactly)
# ─────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    tokens = word_tokenize(text)
    tokens = [
        _lemmatizer.lemmatize(tok)
        for tok in tokens
        if tok not in STOP_WORDS and len(tok) > 2
    ]
    return " ".join(tokens)

# ─────────────────────────────────────────────
# Core prediction logic
# ─────────────────────────────────────────────
def _get_confidence(model, vec_text) -> float:
    """
    Returns confidence (0–100 %).
    Uses predict_proba if available, else normalised decision_function.
    """
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(vec_text)[0]
            return round(float(max(proba)) * 100, 1)
        else:
            # LinearSVC: use decision_function
            score = model.decision_function(vec_text)[0]
            # Sigmoid normalisation → probability-like value
            confidence = 1 / (1 + np.exp(-abs(score)))
            return round(float(confidence) * 100, 1)
    except Exception:
        return 0.0


def predict_news(text: str) -> dict:
    """
    Runs the full prediction pipeline on raw input text.

    Returns a dict with:
        prediction   : "FAKE" | "REAL"
        confidence   : float (0–100)
        model_used   : str
        timestamp    : str
        snippet      : str  (first 120 chars of raw input)
    """
    if MODEL is None or VECTORIZER is None:
        return {
            "error": (
                "Model not loaded. Please run train.py first to train "
                "and save the model."
            )
        }

    cleaned = preprocess_text(text)

    if len(cleaned.split()) < 3:
        return {
            "error": (
                "The text is too short or contains only common words. "
                "Please provide a news article with at least a few sentences."
            )
        }

    vec_text = VECTORIZER.transform([cleaned])
    raw_pred = MODEL.predict(vec_text)[0]
    label    = "FAKE" if raw_pred == 1 else "REAL"
    conf     = _get_confidence(MODEL, vec_text)

    return {
        "prediction": label,
        "confidence": conf,
        "model_used": BEST_MODEL_NAME,
        "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "snippet":    text[:120].strip(),
    }

# ─────────────────────────────────────────────
# Input validation
# ─────────────────────────────────────────────
MIN_CHARS   = 50
MAX_CHARS   = 10_000


def validate_input(text: str) -> str | None:
    """Returns an error string or None if valid."""
    if not text or not text.strip():
        return "Please enter a news article or headline."
    if len(text.strip()) < MIN_CHARS:
        return f"Text too short ({len(text.strip())} chars). Please enter at least {MIN_CHARS} characters."
    if len(text) > MAX_CHARS:
        return f"Text too long ({len(text):,} chars). Maximum allowed is {MAX_CHARS:,} characters."
    return None

def verify_news(query):
    try:
        url = "https://newsapi.org/v2/everything"

        params = {
            "q": query,
            "language": "en",
            "sortBy": "relevancy",
            "pageSize": 5,
            "apiKey": NEWS_API_KEY
        }

        response = requests.get(url, params=params)
        data = response.json()

        articles = []

        for article in data.get("articles", []):
            articles.append({
                "source": article["source"]["name"],
                "title": article["title"],
                "url": article["url"]
            })

        return {
            "count": data.get("totalResults", 0),
            "articles": articles
        }

    except Exception as e:
        return {
            "count": 0,
            "articles": [],
            "error": str(e)
        }    

# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    """Home page with the prediction form."""
    model_ready = MODEL is not None and VECTORIZER is not None
    return render_template(
        "index.html",
        model_ready=model_ready,
        best_model_name=BEST_MODEL_NAME,
        metrics=METRICS,
        history=_prediction_history[-10:][::-1],   # latest 10, newest first
    )


@app.route("/predict", methods=["POST"])
def predict():
    """POST — accepts news text, returns prediction, redirects to index."""
    global _pred_id_counter

    news_text = request.form.get("news_text", "").strip()
    error = validate_input(news_text)

    if error:
        return render_template(
            "index.html",
            error=error,
            news_text=news_text,
            model_ready=(MODEL is not None),
            best_model_name=BEST_MODEL_NAME,
            metrics=METRICS,
            history=_prediction_history[-10:][::-1],
        )

    result = predict_news(news_text)

    verification = verify_news(news_text.split(".")[0])
    
    print("Verification Results:", verification)

    result["verification_count"] = verification["count"]
    result["verification_articles"] = verification["articles"]

    if "error" in result:
        return render_template(
            "index.html",
            error=result["error"],
            news_text=news_text,
            model_ready=(MODEL is not None),
            best_model_name=BEST_MODEL_NAME,
            metrics=METRICS,
            history=_prediction_history[-10:][::-1],
        )

    # Persist prediction
    _pred_id_counter += 1
    record = {
        "id":         _pred_id_counter,
        "timestamp":  result["timestamp"],
        "snippet":    result["snippet"],
        "full_text":  news_text[:500],
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "model_used": result["model_used"],
    }
    _prediction_history.append(record)
    if len(_prediction_history) > MAX_HISTORY:
        _prediction_history.pop(0)
    _append_history(record)

    return render_template(
        "index.html",
        result=result,
        news_text=news_text,
        model_ready=True,
        best_model_name=BEST_MODEL_NAME,
        metrics=METRICS,
        history=_prediction_history[-10:][::-1],
    )


@app.route("/dashboard")
def dashboard():
    """Analytics dashboard."""
    # Aggregate stats from history
    total     = len(_prediction_history)
    fake_cnt  = sum(1 for r in _prediction_history if r["prediction"] == "FAKE")
    real_cnt  = total - fake_cnt
    avg_conf  = (
        round(sum(r["confidence"] for r in _prediction_history) / total, 1)
        if total > 0 else 0
    )

    stats = {
        "total":     total,
        "fake":      fake_cnt,
        "real":      real_cnt,
        "avg_conf":  avg_conf,
        "fake_pct":  round(fake_cnt / total * 100, 1) if total > 0 else 0,
        "real_pct":  round(real_cnt / total * 100, 1) if total > 0 else 0,
    }

    # Check which EDA images exist
    images = {}
    image_map = {
        "label_dist":    "static/images/label_distribution.png",
        "article_len":   "static/images/article_length.png",
        "top_words":     "static/images/top_words.png",
        "wc_fake":       "static/images/wordcloud_fake.png",
        "wc_real":       "static/images/wordcloud_real.png",
        "model_compare": "static/images/model_comparison.png",
    }
    for key, path in image_map.items():
        images[key] = os.path.exists(path)

    return render_template(
        "dashboard.html",
        stats=stats,
        metrics=METRICS,
        best_model_name=BEST_MODEL_NAME,
        history=_prediction_history[-20:][::-1],
        images=images,
    )


# ─────────────────────────────────────────────
# JSON API endpoints (for JS charts)
# ─────────────────────────────────────────────
@app.route("/api/history")
def api_history():
    limit = min(int(request.args.get("limit", 50)), 200)
    return jsonify(_prediction_history[-limit:][::-1])


@app.route("/api/stats")
def api_stats():
    total    = len(_prediction_history)
    fake_cnt = sum(1 for r in _prediction_history if r["prediction"] == "FAKE")
    real_cnt = total - fake_cnt
    return jsonify({
        "total": total,
        "fake":  fake_cnt,
        "real":  real_cnt,
        "metrics": METRICS,
        "best_model": BEST_MODEL_NAME,
    })


# ─────────────────────────────────────────────
# Error handlers
# ─────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return render_template("index.html",
                           error="Page not found (404).",
                           model_ready=(MODEL is not None),
                           best_model_name=BEST_MODEL_NAME,
                           metrics=METRICS,
                           history=[]), 404


@app.errorhandler(500)
def server_error(e):
    log.error(f"Internal server error: {e}")
    return render_template("index.html",
                           error="Internal server error. Please try again.",
                           model_ready=(MODEL is not None),
                           best_model_name=BEST_MODEL_NAME,
                           metrics=METRICS,
                           history=[]), 500


@app.errorhandler(413)
def too_large(e):
    return render_template("index.html",
                           error="Input too large. Maximum 2 MB accepted.",
                           model_ready=(MODEL is not None),
                           best_model_name=BEST_MODEL_NAME,
                           metrics=METRICS,
                           history=[]), 413


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    log.info(f"Starting Fake News Detection server on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug)