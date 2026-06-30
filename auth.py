"""
Thin wrapper around the Supabase Python client for auth + history.
"""
from __future__ import annotations
import os
from typing import Any, Dict, List
from supabase import create_client, Client

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    raise RuntimeError(
        "SUPABASE_URL and SUPABASE_ANON_KEY must be set in your .env file."
    )

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)


class AuthError(Exception):
    """Raised when Supabase rejects a signup/login attempt."""
    pass


def sign_up(email: str, password: str) -> Dict[str, Any]:
    try:
        result = supabase.auth.sign_up({"email": email, "password": password})
    except Exception as exc:
        raise AuthError(_clean_error(exc)) from exc

    if result.user is None:
        raise AuthError("Sign up failed. Please try again.")

    if result.session is None:
        # Email confirmation is still enabled in Supabase — turn it off
        # in Authentication > Providers > Email > Confirm email.
        raise AuthError(
            "Account created but email confirmation is required. "
            "Please contact support."
        )

    return {
        "id": result.user.id,
        "email": result.user.email,
        "access_token": result.session.access_token,
    }


def sign_in(email: str, password: str) -> Dict[str, Any]:
    try:
        result = supabase.auth.sign_in_with_password(
            {"email": email, "password": password}
        )
    except Exception as exc:
        raise AuthError(_clean_error(exc)) from exc

    if result.user is None:
        raise AuthError("Invalid email or password.")

    access_token = result.session.access_token if result.session else None
    return {"id": result.user.id, "email": result.user.email, "access_token": access_token}


def sign_out() -> None:
    try:
        supabase.auth.sign_out()
    except Exception:
        pass


def send_password_reset(email: str, redirect_url: str | None = None) -> None:
    """
    Sends a password reset email via Supabase.
    The user will receive a link that redirects to redirect_url with
    recovery tokens in the URL fragment (#access_token=...&type=recovery).
    Raises AuthError on failure.
    """
    try:
        options = {}
        if redirect_url:
            options["redirect_to"] = redirect_url
        supabase.auth.reset_password_email(email, options)
    except Exception as exc:
        raise AuthError(_clean_error(exc)) from exc


def update_password(access_token: str, refresh_token: str | None, new_password: str) -> None:
    """
    Updates the user's password using their recovery tokens.
    Called after the user clicks the reset link in their email and
    submits a new password on the reset page.
    Raises AuthError if the tokens are invalid or expired.
    """
    try:
        client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
        # Set the recovery session so Supabase knows who is resetting
        client.auth.set_session(access_token, refresh_token or "")
        client.auth.update_user({"password": new_password})
    except Exception as exc:
        raise AuthError(
            "Could not update your password. Your reset link may have expired — "
            "please request a new one."
        ) from exc


def _client_for(access_token: str | None) -> Client:
    client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    if access_token:
        client.postgrest.auth(access_token)
    return client


def save_prediction(
    user_id: str, email_text: str, label: str, confidence: float, access_token: str | None = None
) -> None:
    client = _client_for(access_token)
    client.table("predictions").insert(
        {
            "user_id": user_id,
            "email_text": email_text,
            "label": label,
            "confidence": confidence,
        }
    ).execute()


def get_history(user_id: str, access_token: str | None = None, limit: int = 50) -> List[Dict[str, Any]]:
    client = _client_for(access_token)
    response = (
        client.table("predictions")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return response.data or []


def update_prediction(
    prediction_id: str,
    user_id: str,
    email_text: str,
    label: str,
    confidence: float,
    access_token: str | None = None,
) -> None:
    client = _client_for(access_token)
    client.table("predictions").update(
        {
            "email_text": email_text,
            "label": label,
            "confidence": confidence,
        }
    ).eq("id", prediction_id).eq("user_id", user_id).execute()


def delete_prediction(prediction_id: str, user_id: str, access_token: str | None = None) -> None:
    client = _client_for(access_token)
    client.table("predictions").delete().eq("id", prediction_id).eq("user_id", user_id).execute()


def delete_all_predictions(user_id: str, access_token: str | None = None) -> None:
    client = _client_for(access_token)
    client.table("predictions").delete().eq("user_id", user_id).execute()


def _clean_error(exc: Exception) -> str:
    message = str(exc)
    msg = message.lower()
    if "already registered" in msg:
        return "An account with that email already exists. Try logging in instead."
    if "invalid login credentials" in msg or "invalid email or password" in msg:
        return "Incorrect email or password. Please try again."
    if "email not confirmed" in msg:
        return "Please confirm your email before logging in."
    if "password" in msg and "short" in msg:
        return "Password is too short. Use at least 6 characters."
    if "banned" in msg:
        return "This account has been restricted."
    if "rate limit" in msg:
        return "Too many attempts. Please wait a moment and try again."
    return "Something went wrong. Please try again."