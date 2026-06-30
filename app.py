from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
from functools import wraps

from flask import (
    Flask, render_template, request, redirect, url_for, session, flash, jsonify
)
from dotenv import load_dotenv
import joblib
import numpy as np
import os

load_dotenv()  # reads .env into os.environ before anything else runs

import auth  # our Supabase wrapper (auth.py)


app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-fallback-key-change-me")


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


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def current_user() -> Dict[str, Any] | None:
    """Returns the logged-in user's session info, or None if logged out."""
    return session.get("user")


def login_required(view_func):
    """Decorator for routes that require a logged-in user."""
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if current_user() is None:
            flash("Please log in to view that page.")
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)
    return wrapped


# Make `current_user()` available inside every template as `user`,
# so base.html can show "Log in / Sign up" vs the profile pill.
@app.context_processor
def inject_user():
    return {"user": current_user()}


# ---------------------------------------------------------------------------
# Core routes
# ---------------------------------------------------------------------------

@app.get("/")
def home() -> str:
    return render_template("index.html")


@app.post("/predict")
def predict() -> str:
    email_text = request.form.get("email_text", "")
    pred = predict_text(email_text)

    # Only save to history if someone is actually logged in.
    user = current_user()
    if user is not None and email_text.strip():
        try:
            auth.save_prediction(
                user_id=user["id"],
                email_text=email_text,
                label=pred.label,
                confidence=pred.confidence,
                access_token=user.get("access_token"),
            )
        except Exception:
            # Don't let a history-save failure break the prediction result
            # the user is waiting on. Could log this to a real logger later.
            pass

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


# ---------------------------------------------------------------------------
# Auth routes (email + password only)
# ---------------------------------------------------------------------------

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("login.html")

    email = request.form.get("email", "").strip()
    password = request.form.get("password", "")

    try:
        user = auth.sign_in(email, password)
    except auth.AuthError as e:
        flash(str(e))
        return render_template("login.html"), 400

    session["user"] = {
        "id": user["id"],
        "email": user["email"],
        "access_token": user["access_token"],
    }
    return redirect(url_for("home"))


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "GET":
        return render_template("signup.html")

    email = request.form.get("email", "").strip()
    password = request.form.get("password", "")

    try:
        user = auth.sign_up(email, password)
    except auth.AuthError as e:
        flash(str(e))
        return render_template("signup.html"), 400

    session["user"] = {
        "id": user["id"],
        "email": user["email"],
        "access_token": user["access_token"],
    }
    return redirect(url_for("home"))


@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    return render_template("forgot_password.html")


@app.get("/logout")
def logout():
    auth.sign_out()
    session.clear()
    return redirect(url_for("home"))


# ---------------------------------------------------------------------------
# History (JSON API consumed by the modal on index.html)
# ---------------------------------------------------------------------------

@app.get("/api/history")
def api_history():
    """
    Returns the logged-in user's prediction history as JSON.
    The History tab modal on the home page calls this via fetch()
    to decide whether to show the sign-in prompt or the real list.
    """
    user = current_user()
    if user is None:
        return jsonify({"logged_in": False, "predictions": []})

    try:
        rows = auth.get_history(user["id"], access_token=user.get("access_token"))
    except Exception as exc:
        app.logger.exception("api_history failed for user %s", user.get("id"))
        return jsonify({"logged_in": True, "predictions": [], "error": True})

    return jsonify({"logged_in": True, "predictions": rows})


@app.post("/api/history/<prediction_id>")
def api_history_update(prediction_id: str):
    """
    Re-runs the model on edited email text and overwrites that one
    history row. Mirrors what /recalculate does on the results page,
    but persists the new result instead of just rendering it once.
    """
    user = current_user()
    if user is None:
        return jsonify({"ok": False, "error": "Not logged in."}), 401

    email_text = request.form.get("email_text", "") or (request.get_json(silent=True) or {}).get("email_text", "")
    pred = predict_text(email_text)

    try:
        auth.update_prediction(
            prediction_id=prediction_id,
            user_id=user["id"],
            email_text=email_text,
            label=pred.label,
            confidence=pred.confidence,
            access_token=user.get("access_token"),
        )
    except Exception:
        app.logger.exception("api_history_update failed for prediction %s", prediction_id)
        return jsonify({"ok": False, "error": "Could not update that entry."}), 500

    return jsonify({
        "ok": True,
        "prediction": {
            "id": prediction_id,
            "email_text": email_text,
            "label": pred.label,
            "confidence": pred.confidence,
        },
    })


@app.delete("/api/history/<prediction_id>")
def api_history_delete(prediction_id: str):
    """Deletes a single history entry belonging to the logged-in user."""
    user = current_user()
    if user is None:
        return jsonify({"ok": False, "error": "Not logged in."}), 401

    try:
        auth.delete_prediction(
            prediction_id=prediction_id,
            user_id=user["id"],
            access_token=user.get("access_token"),
        )
    except Exception:
        app.logger.exception("api_history_delete failed for prediction %s", prediction_id)
        return jsonify({"ok": False, "error": "Could not delete that entry."}), 500

    return jsonify({"ok": True})


@app.delete("/api/history")
def api_history_delete_all():
    """Deletes every history entry belonging to the logged-in user."""
    user = current_user()
    if user is None:
        return jsonify({"ok": False, "error": "Not logged in."}), 401

    try:
        auth.delete_all_predictions(user_id=user["id"], access_token=user.get("access_token"))
    except Exception:
        app.logger.exception("api_history_delete_all failed for user %s", user.get("id"))
        return jsonify({"ok": False, "error": "Could not clear history."}), 500

    return jsonify({"ok": True})


if __name__ == "__main__":
    # For local development
    app.run(host="127.0.0.1", port=5000, debug=True)