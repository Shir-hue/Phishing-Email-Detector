from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any

from flask import Flask, render_template, request

import joblib
import numpy as np

app = Flask(__name__)


@dataclass
class Prediction:
    label: str
    confidence: float  # 0-1
    reason: List[str]
    word_count: int
    char_count: int
    

# Load model and vectorizer once when Flask starts
model = joblib.load("models/model.joblib")
vectorizer = joblib.load("models/vectorizer.joblib")


def predict_text(email_text: str) -> Prediction:

    text = (email_text or "").strip()

    word_count = len(text.split())
    char_count = len(text)

    if not text:
        return Prediction(
            label="Legitimate",
            confidence=0.0,
            reason=["No email text provided."],
            word_count=0,
            char_count=0
        )

    # Convert text to TF-IDF
    text_tfidf = vectorizer.transform([text])

    # Prediction
    prediction = model.predict(text_tfidf)[0]

    # Probability
    probabilities = model.predict_proba(text_tfidf)[0]

    confidence = float(np.max(probabilities))

    label = "Phishing" if prediction == 1 else "Legitimate"

    # Suspicious terms
    suspicious_terms = [
        "urgent",
        "verify",
        "password",
        "account",
        "bank",
        "click",
        "login",
        "security",
        "suspended",
        "confirm",
        "update",
        "payment"
    ]

    found_terms = []

    for term in suspicious_terms:
        if term in text.lower():
            found_terms.append(term)

    reason = []

    if found_terms:
        reason.append(
            f"Suspicious terms detected: {', '.join(found_terms[:5])}"
        )
    else:
        reason.append(
            "No common phishing keywords detected."
        )

    reason.append(f"Model confidence: {confidence:.2%}")

    return Prediction(
        label=label,
        confidence=confidence,
        reason=reason,
        word_count=word_count,
        char_count=char_count,
    )


@app.get("/")
def home() -> str:
    return render_template("index.html")


@app.post("/predict")
def predict() -> str:
    email_text = request.form.get("email_text", "")
    pred = predict_text(email_text)

    return render_template(
        "results.html",
        email_text=email_text,
        label=pred.label,
        confidence=pred.confidence,
        reason=pred.reason,
        word_count=pred.word_count,
        char_count=pred.char_count,
    )


@app.post("/recalculate")
def recalculate() -> str:
    email_text = request.form.get("email_text", "")
    pred = predict_text(email_text)

    return render_template(
        "results.html",
        email_text=email_text,
        label=pred.label,
        confidence=pred.confidence,
        reason=pred.reason,
        word_count=pred.word_count,
        char_count=pred.char_count,
    )


@app.get("/about")
def about() -> str:
    return render_template("about.html")


@app.get("/security-advice")
def security_advice() -> str:
    return render_template("security_advice.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    return render_template("signup.html")

@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    return render_template("forgot_password.html")


if __name__ == "__main__":
    # For local development
    app.run(host="127.0.0.1", port=5000, debug=True)

