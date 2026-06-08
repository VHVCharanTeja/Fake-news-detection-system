

import os
import re
import string
import warnings
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import seaborn as sns

"""
train.py — Fake News Detection: Full ML Training Pipeline
==========================================================
Covers:
  Phase 1 : NLP Preprocessing
  Phase 2 : Exploratory Data Analysis (EDA)
  Phase 3 : Feature Engineering (TF-IDF)
  Phase 4 : Model Training & Evaluation
  Phase 5 : Model Selection & Serialisation
"""

from wordcloud import WordCloud
from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)
import joblib

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 0. Directory Setup
# ─────────────────────────────────────────────
DIRS = ["models", "static/images", "static/css", "static/js", "data"]
for d in DIRS:
    os.makedirs(d, exist_ok=True)

# ─────────────────────────────────────────────
# 1. NLTK Downloads
# ─────────────────────────────────────────────
def download_nltk_data():
    packages = ["stopwords", "wordnet", "punkt", "omw-1.4", "punkt_tab"]
    for pkg in packages:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass

download_nltk_data()

STOP_WORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ─────────────────────────────────────────────
# 2. Load Dataset
# ─────────────────────────────────────────────
def load_data(fake_path: str = "data/Fake.csv",
              true_path: str = "data/True.csv") -> pd.DataFrame:
    """
    Loads Fake.csv and True.csv, adds a binary label column,
    and returns a single shuffled DataFrame.

    Label encoding:
        0  →  Real news
        1  →  Fake news
    """
    log.info("Loading datasets …")

    if not os.path.exists(fake_path):
        raise FileNotFoundError(f"Cannot find {fake_path}. Place Fake.csv inside the data/ folder.")
    if not os.path.exists(true_path):
        raise FileNotFoundError(f"Cannot find {true_path}. Place True.csv inside the data/ folder.")

    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    fake_df["label"] = 1   # 1 = Fake
    true_df["label"] = 0   # 0 = Real

    df = pd.concat([fake_df, true_df], ignore_index=True)

    # Combine title + text for richer signal
    text_col = None
    for col in ["text", "content", "body"]:
        if col in df.columns:
            text_col = col
            break

    title_col = "title" if "title" in df.columns else None

    if text_col and title_col:
        df["content"] = df[title_col].fillna("") + " " + df[text_col].fillna("")
    elif text_col:
        df["content"] = df[text_col].fillna("")
    elif title_col:
        df["content"] = df[title_col].fillna("")
    else:
        raise ValueError("Dataset must contain a 'text' or 'title' column.")

    df = df[["content", "label"]].dropna().sample(frac=1, random_state=42).reset_index(drop=True)

    log.info(f"  Total samples : {len(df):,}")
    log.info(f"  Fake news     : {(df['label']==1).sum():,}")
    log.info(f"  Real news     : {(df['label']==0).sum():,}")
    return df

# ─────────────────────────────────────────────
# 3. NLP Preprocessing  (Phase 1)
# ─────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    """
    Full NLP preprocessing pipeline applied to each article.

    Steps (in order):
    1. Lowercase       — normalises case so 'News' == 'news'
    2. URL removal     — URLs carry no semantic meaning
    3. HTML tags       — strip leftover markup
    4. Punctuation     — punctuation doesn't contribute to topic modelling
    5. Special chars   — removes @, #, $, etc.
    6. Numbers         — digits are rarely informative for fake-news detection
    7. Tokenisation    — split into individual words for per-token operations
    8. Stopword removal— removes high-frequency but low-information words
    9. Lemmatisation   — reduces inflected forms to root (running → run)
   10. Short token filter — removes 1-2 char tokens that are noise
    """

    # Step 1 — Lowercase
    text = text.lower()

    # Step 2 — Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Step 3 — Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Step 4 — Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Step 5 — Remove special characters (keep only a-z and spaces)
    text = re.sub(r"[^a-z\s]", " ", text)

    # Step 6 — Remove numbers
    text = re.sub(r"\d+", " ", text)

    # Step 7 — Tokenise
    tokens = word_tokenize(text)

    # Steps 8, 9, 10 — Stopword removal, Lemmatisation, short token filter
    tokens = [
        lemmatizer.lemmatize(tok)
        for tok in tokens
        if tok not in STOP_WORDS and len(tok) > 2
    ]

    return " ".join(tokens)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Running NLP preprocessing (this may take a minute) …")
    df = df.copy()
    df["clean_content"] = df["content"].apply(preprocess_text)
    df["word_count"] = df["clean_content"].apply(lambda x: len(x.split()))
    log.info("Preprocessing complete.")
    return df

# ─────────────────────────────────────────────
# 4. EDA  (Phase 2)
# ─────────────────────────────────────────────
PALETTE = {"Fake": "#E74C3C", "Real": "#2ECC71"}

def set_plot_style():
    sns.set_theme(style="darkgrid", palette="muted")
    plt.rcParams.update({
        "figure.facecolor": "#1a1a2e",
        "axes.facecolor":   "#16213e",
        "axes.labelcolor":  "#e0e0e0",
        "xtick.color":      "#e0e0e0",
        "ytick.color":      "#e0e0e0",
        "text.color":       "#e0e0e0",
        "grid.color":       "#2d2d5e",
        "axes.titlecolor":  "#ffffff",
    })

set_plot_style()


def plot_label_distribution(df: pd.DataFrame):
    """Bar chart — Fake vs Real news count."""
    counts = df["label"].value_counts()
    labels = ["Real News", "Fake News"]
    colors = [PALETTE["Real"], PALETTE["Fake"]]
    values = [counts.get(0, 0), counts.get(1, 0)]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="#ffffff22", width=0.5)
    ax.set_title("Fake vs Real News Distribution", fontsize=16, fontweight="bold", pad=15)
    ax.set_ylabel("Article Count", fontsize=12)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 200,
                f"{val:,}", ha="center", va="bottom", fontsize=12, color="white")
    plt.tight_layout()
    plt.savefig("static/images/label_distribution.png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    log.info("Saved label_distribution.png")


def plot_article_length(df: pd.DataFrame):
    """KDE plot — word-count distribution per class."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for label_val, label_name, color in [(0, "Real", PALETTE["Real"]),
                                          (1, "Fake", PALETTE["Fake"])]:
        subset = df[df["label"] == label_val]["word_count"]
        sns.kdeplot(subset, ax=ax, label=label_name, color=color, fill=True, alpha=0.4)
    ax.set_title("Article Length Distribution (Word Count)", fontsize=15, fontweight="bold")
    ax.set_xlabel("Word Count", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig("static/images/article_length.png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    log.info("Saved article_length.png")


def get_top_words(df: pd.DataFrame, label_val: int, n: int = 20) -> list:
    """Return top-n (word, count) tuples for a given label."""
    corpus = " ".join(df[df["label"] == label_val]["clean_content"].tolist())
    counter = Counter(corpus.split())
    return counter.most_common(n)


def plot_top_words(df: pd.DataFrame):
    """Horizontal bar chart — most common words for each class."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, label_val, label_name, color in [
        (axes[0], 0, "Real News", PALETTE["Real"]),
        (axes[1], 1, "Fake News", PALETTE["Fake"]),
    ]:
        top = get_top_words(df, label_val, 15)
        words, counts = zip(*top)
        bars = ax.barh(words[::-1], counts[::-1], color=color, edgecolor="#ffffff11")
        ax.set_title(f"Top 15 Words — {label_name}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Frequency", fontsize=11)
        for bar, count in zip(bars, counts[::-1]):
            ax.text(bar.get_width() + max(counts) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{count:,}", va="center", fontsize=9, color="white")
    plt.suptitle("Most Common Words by Category", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("static/images/top_words.png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    log.info("Saved top_words.png")


def generate_wordcloud(df: pd.DataFrame, label_val: int, label_name: str):
    """WordCloud for a given class label."""
    corpus = " ".join(df[df["label"] == label_val]["clean_content"].tolist())
    bg_color = "#1a1a2e"
    colormap = "Reds" if label_val == 1 else "Greens"

    wc = WordCloud(
        width=900, height=500,
        background_color=bg_color,
        colormap=colormap,
        max_words=200,
        collocations=False,
        prefer_horizontal=0.7,
    ).generate(corpus)

    fig, ax = plt.subplots(figsize=(11, 5), facecolor=bg_color)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"Word Cloud — {label_name} News", fontsize=16,
                 fontweight="bold", color="white", pad=12)
    fname = f"static/images/wordcloud_{'fake' if label_val == 1 else 'real'}.png"
    plt.savefig(fname, dpi=130, bbox_inches="tight", facecolor=bg_color)
    plt.close()
    log.info(f"Saved {fname}")


def run_eda(df: pd.DataFrame):
    log.info("Running EDA …")
    plot_label_distribution(df)
    plot_article_length(df)
    plot_top_words(df)
    generate_wordcloud(df, 1, "Fake")
    generate_wordcloud(df, 0, "Real")
    log.info("EDA complete — charts saved to static/images/")

# ─────────────────────────────────────────────
# 5. Feature Engineering  (Phase 3)
# ─────────────────────────────────────────────
def build_tfidf_features(X_train, X_test):
    """
    TF-IDF Vectorizer with unigrams + bigrams.

    Why TF-IDF over CountVectorizer?
    ─────────────────────────────────
    • CountVectorizer gives raw frequencies, heavily biased toward common words.
    • TF-IDF penalises words that appear in many documents (low IDF) while
      rewarding words that are frequent in specific documents (high TF).
    • This makes it much better at capturing *distinguishing* vocabulary
      (e.g. politically charged terms unique to fake vs real articles).
    • Bigrams (word pairs) capture phrases like "fake news", "white house",
      which carry more context than individual words alone.
    • max_features=50000 limits the vocabulary to control overfitting and
      keeps training time reasonable.
    """
    log.info("Building TF-IDF features (unigrams + bigrams, max 50k) …")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=50_000,
        sublinear_tf=True,       # applies 1+log(tf) dampening
        min_df=2,                # ignore rare terms
        max_df=0.95,             # ignore near-ubiquitous terms
        strip_accents="unicode",
        analyzer="word",
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)
    log.info(f"  Feature matrix shape: {X_train_tfidf.shape}")
    return X_train_tfidf, X_test_tfidf, vectorizer

# ─────────────────────────────────────────────
# 6. Model Training & Evaluation  (Phase 4)
# ─────────────────────────────────────────────
MODELS = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, C=1.0, solver="lbfgs", random_state=42
    ),
    "Linear SVM": LinearSVC(
        C=1.0, max_iter=2000, random_state=42
    ),
    "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
}


def evaluate_model(name, model, X_train, X_test, y_train, y_test) -> dict:
    """Train one model and return a metrics dict."""
    log.info(f"  Training: {name} …")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "model":     name,
        "accuracy":  round(accuracy_score(y_test, y_pred) * 100, 2),
        "precision": round(precision_score(y_test, y_pred) * 100, 2),
        "recall":    round(recall_score(y_test, y_pred) * 100, 2),
        "f1":        round(f1_score(y_test, y_pred) * 100, 2),
    }

    log.info(
        f"    Acc={metrics['accuracy']}%  "
        f"P={metrics['precision']}%  "
        f"R={metrics['recall']}%  "
        f"F1={metrics['f1']}%"
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Real", "Fake"],
                yticklabels=["Real", "Fake"], ax=ax,
                linewidths=0.5, linecolor="#ffffff22")
    ax.set_title(f"Confusion Matrix — {name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    plt.tight_layout()
    safe_name = name.lower().replace(" ", "_")
    plt.savefig(f"static/images/cm_{safe_name}.png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()

    return metrics, model


def plot_model_comparison(results: list):
    """Grouped bar chart comparing all model metrics."""
    metrics_keys = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(results))
    width = 0.18
    colors = ["#3498db", "#9b59b6", "#e67e22", "#1abc9c"]
    labels = ["Accuracy", "Precision", "Recall", "F1 Score"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (metric, label, color) in enumerate(zip(metrics_keys, labels, colors)):
        values = [r[metric] for r in results]
        bars = ax.bar(x + i * width, values, width, label=label, color=color, alpha=0.88)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=7.5, color="white")

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([r["model"] for r in results], fontsize=11)
    ax.set_ylim(85, 102)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("Model Comparison — All Metrics", fontsize=15, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("static/images/model_comparison.png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    log.info("Saved model_comparison.png")


def train_all_models(X_train, X_test, y_train, y_test):
    log.info("Training all models …")
    results = []
    trained_models = {}

    for name, model in MODELS.items():
        metrics, trained = evaluate_model(
            name, model, X_train, X_test, y_train, y_test
        )
        results.append(metrics)
        trained_models[name] = trained

    # Print comparison table
    df_results = pd.DataFrame(results).set_index("model")
    print("\n" + "=" * 60)
    print("MODEL COMPARISON TABLE")
    print("=" * 60)
    print(df_results.to_string())
    print("=" * 60 + "\n")

    plot_model_comparison(results)
    return results, trained_models

# ─────────────────────────────────────────────
# 7. Model Selection & Save  (Phase 5)
# ─────────────────────────────────────────────
def select_and_save(results: list, trained_models: dict, vectorizer):
    """Pick the model with the highest F1 score and serialise it."""
    best = max(results, key=lambda r: r["f1"])
    best_name = best["model"]
    best_model = trained_models[best_name]

    log.info(f"Best model: {best_name}  (F1={best['f1']}%)")

    joblib.dump(best_model, "models/model.pkl")
    joblib.dump(vectorizer,  "models/vectorizer.pkl")

    # Save metrics for the Flask app to display
    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv("models/metrics.csv", index=False)

    # Save best model name for Flask
    with open("models/best_model_name.txt", "w") as f:
        f.write(best_name)

    log.info("Saved model.pkl, vectorizer.pkl, metrics.csv, best_model_name.txt → models/")
    return best_name, best

# ─────────────────────────────────────────────
# 8. Main Orchestrator
# ─────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("  FAKE NEWS DETECTION — TRAINING PIPELINE")
    print("=" * 60 + "\n")

    # Phase 1 + 2: Load & preprocess
    df = load_data()
    df = preprocess_dataframe(df)

    # Phase 2: EDA
    run_eda(df)

    # Phase 3: TF-IDF
    X = df["clean_content"]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_tfidf, X_test_tfidf, vectorizer = build_tfidf_features(X_train, X_test)

    # Phase 4: Train & evaluate
    results, trained_models = train_all_models(
        X_train_tfidf, X_test_tfidf, y_train, y_test
    )

    # Phase 5: Select & save
    best_name, best_metrics = select_and_save(results, trained_models, vectorizer)

    print("\n✅ Training complete!")
    print(f"   Best Model : {best_name}")
    print(f"   Accuracy   : {best_metrics['accuracy']}%")
    print(f"   F1 Score   : {best_metrics['f1']}%")
    print("\n   Run `python app.py` to start the web app.\n")


if __name__ == "__main__":
    main()