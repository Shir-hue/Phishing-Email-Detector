from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
from functools import wraps
from datetime import datetime, timezone

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

# Comma-separated list of emails that get access to /admin.
# Set ADMIN_EMAILS in Render's environment variables — never ever hardcode it here.
# Example:  ADMIN_EMAILS=you@example.com,partner@example.com
ADMIN_EMAILS = {
    e.strip().lower()
    for e in os.environ.get("ADMIN_EMAILS", "").split(",")
    if e.strip()
}


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
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)[0]
    probabilities = model.predict_proba(text_tfidf)[0]
    confidence = float(np.max(probabilities))
    label = "Phishing" if prediction == 1 else "Legitimate"
    suspicious_terms = [
        "urgent", "verify", "password", "account", "bank",
        "click", "login", "security", "suspended", "confirm",
        "update", "payment"
    ]
    found_terms = [t for t in suspicious_terms if t in text.lower()]
    reason = []
    if found_terms:
        reason.append(f"Suspicious terms detected: {', '.join(found_terms[:5])}")
    else:
        reason.append("No common phishing keywords detected.")
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
    return session.get("user")


def is_admin(user: Dict[str, Any] | None = None) -> bool:
    """True if the given user (or the currently logged-in user) is an admin."""
    user = user if user is not None else current_user()
    if user is None:
        return False
    return user.get("email", "").lower() in ADMIN_EMAILS


def login_required(view_func):
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if current_user() is None:
            flash("Please log in to view that page.")
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)
    return wrapped


def admin_required(view_func):
    """Decorator for routes that require a logged-in admin."""
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        user = current_user()
        if user is None:
            flash("Please log in to view that page.")
            return redirect(url_for("login"))
        if not is_admin(user):
            flash("You don't have access to that page.")
            return redirect(url_for("home"))
        return view_func(*args, **kwargs)
    return wrapped


# Makes `user` and `user_is_admin` available in every template automatically.
# base.html uses user_is_admin to show/hide the "Admin panel" dropdown link.
@app.context_processor
def inject_user():
    user = current_user()
    return {"user": user, "user_is_admin": is_admin(user)}


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
        return render_template("login.html", reset_success=request.args.get("reset") == "1")

    email = request.form.get("email", "").strip()
    password = request.form.get("password", "")

    try:
        user = auth.sign_in(email, password)
    except auth.AuthError as e:
        app.logger.exception("Login failed for %s", email)
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
        app.logger.exception("Signup failed for %s", email)
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
    if request.method == "GET":
        return render_template("forgot_password.html")

    email = request.form.get("email", "").strip()
    if email:
        try:
            reset_url = url_for("auth_reset", _external=True)
            auth.send_password_reset(email, redirect_url=reset_url)
        except Exception:
            app.logger.exception("Password reset failed for %s", email)

    return render_template("forgot_password.html", email_sent=True, email=email)


@app.get("/logout")
def logout():
    auth.sign_out()
    session.clear()
    return redirect(url_for("home"))


# ---------------------------------------------------------------------------
# Password reset handshake
# ---------------------------------------------------------------------------

@app.get("/auth/reset")
def auth_reset():
    return render_template("reset_password.html")


@app.post("/auth/reset/finish")
def auth_reset_finish():
    data = request.get_json(silent=True) or {}
    access_token = data.get("access_token")
    refresh_token = data.get("refresh_token")
    new_password = data.get("password", "").strip()

    if not access_token or not new_password:
        return jsonify({"ok": False, "error": "Missing required fields."}), 400
    if len(new_password) < 6:
        return jsonify({"ok": False, "error": "Password must be at least 6 characters."}), 400

    try:
        auth.update_password(access_token, refresh_token, new_password)
    except auth.AuthError as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    return jsonify({"ok": True, "redirect": url_for("login", reset="1")})


# ---------------------------------------------------------------------------
# History (JSON API consumed by the modal on index.html)
# ---------------------------------------------------------------------------

@app.get("/api/history")
def api_history():
    user = current_user()
    if user is None:
        return jsonify({"logged_in": False, "predictions": []})

    try:
        rows = auth.get_history(user["id"], access_token=user.get("access_token"))
    except Exception:
        app.logger.exception("api_history failed for user %s", user.get("id"))
        return jsonify({"logged_in": True, "predictions": [], "error": True})

    return jsonify({"logged_in": True, "predictions": rows})


@app.post("/api/history/<prediction_id>")
def api_history_update(prediction_id: str):
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
    user = current_user()
    if user is None:
        return jsonify({"ok": False, "error": "Not logged in."}), 401

    try:
        auth.delete_all_predictions(user_id=user["id"], access_token=user.get("access_token"))
    except Exception:
        app.logger.exception("api_history_delete_all failed for user %s", user.get("id"))
        return jsonify({"ok": False, "error": "Could not clear history."}), 500

    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Admin panel
# ---------------------------------------------------------------------------

@app.get("/admin")
@admin_required
def admin_dashboard():
    try:
        users = auth.list_users()
        total_predictions = auth.total_prediction_count()
    except auth.AdminError as e:
        flash(str(e))
        users = []
        total_predictions = 0

    today = datetime.now(timezone.utc).date()
    signups_today = 0
    for u in users:
        try:
            created = datetime.fromisoformat(u["created_at"].replace("Z", "+00:00")).date()
            if created == today:
                signups_today += 1
        except Exception:
            pass

    stats = {
        "total_users": len(users),
        "signups_today": signups_today,
        "total_predictions": total_predictions,
    }
    return render_template("admin/dashboard.html", stats=stats)


@app.get("/admin/users")
@admin_required
def admin_users():
    try:
        users = auth.list_users()
    except auth.AdminError as e:
        flash(str(e))
        users = []
    return render_template("admin/users.html", users=users)


@app.post("/admin/users/<user_id>/ban")
@admin_required
def admin_ban_user(user_id: str):
    try:
        auth.ban_user(user_id)
    except auth.AdminError as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    return jsonify({"ok": True})


@app.post("/admin/users/<user_id>/unban")
@admin_required
def admin_unban_user(user_id: str):
    try:
        auth.unban_user(user_id)
    except auth.AdminError as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    return jsonify({"ok": True})


@app.delete("/admin/users/<user_id>")
@admin_required
def admin_delete_user(user_id: str):
    user = current_user()
    if user and user.get("id") == user_id:
        return jsonify({"ok": False, "error": "You can't delete your own admin account."}), 400
    try:
        auth.delete_user(user_id)
    except auth.AdminError as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)