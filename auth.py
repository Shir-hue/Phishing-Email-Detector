"""
Thin wrapper around the Supabase Python client for auth + history.

Keeping this in its own file means app.py doesn't get cluttered with
Supabase-specific error handling, and if you ever swap auth providers
later, this is the only file that has to change.
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
    """
    Creates a new Supabase user with email + password.
    Returns a dict with the user's id/email/access_token on success.
    Raises AuthError with a human-readable message on failure.
    """
    try:
        result = supabase.auth.sign_up({"email": email, "password": password})
    except Exception as exc:
        raise AuthError(_clean_error(exc)) from exc

    if result.user is None:
        raise AuthError("Sign up failed. Please try again.")

    access_token = result.session.access_token if result.session else None
    return {"id": result.user.id, "email": result.user.email, "access_token": access_token}


def sign_in(email: str, password: str) -> Dict[str, Any]:
    """
    Logs in an existing user with email + password.
    Returns a dict with the user's id/email/access_token on success.
    Raises AuthError with a human-readable message on failure.
    """
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
    """Best-effort sign out on the Supabase side."""
    try:
        supabase.auth.sign_out()
    except Exception:
        # Not critical if this fails — we clear the Flask session
        # regardless, which is what actually logs the user out of
        # *this app*.
        pass


def _client_for(access_token: str | None) -> Client:
    """
    Builds a Supabase client scoped to one user's request.

    Why this matters: the module-level `supabase` client is shared
    across every request in the Flask process. If we used it directly
    for database calls, Postgres would see every request as coming
    from the anonymous (anon) role, since login state isn't global -
    it's per-request. Row Level Security policies check auth.uid(),
    which only resolves correctly when the user's own access token is
    attached to the client making the call. So for any call that needs
    to act "as" a specific user, we create a short-lived client and set
    its session to that user's token first.
    """
    client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    if access_token:
        client.postgrest.auth(access_token)
    return client


def save_prediction(
    user_id: str, email_text: str, label: str, confidence: float, access_token: str | None = None
) -> None:
    """Inserts one row into the predictions table for this user."""
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
    """Fetches this user's past predictions, newest first."""
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
    """
    Overwrites one existing history row with re-analyzed text/label/
    confidence. Scoped to user_id so RLS (and this extra check) make
    sure nobody can edit a row that isn't theirs.
    """
    client = _client_for(access_token)
    client.table("predictions").update(
        {
            "email_text": email_text,
            "label": label,
            "confidence": confidence,
        }
    ).eq("id", prediction_id).eq("user_id", user_id).execute()


def delete_prediction(prediction_id: str, user_id: str, access_token: str | None = None) -> None:
    """Deletes one history row, scoped to the owning user."""
    client = _client_for(access_token)
    client.table("predictions").delete().eq("id", prediction_id).eq("user_id", user_id).execute()


def delete_all_predictions(user_id: str, access_token: str | None = None) -> None:
    """Deletes every history row belonging to this user."""
    client = _client_for(access_token)
    client.table("predictions").delete().eq("user_id", user_id).execute()


def _clean_error(exc: Exception) -> str:
    """
    Supabase errors are often verbose/technical. This pulls out a
    short message safe to show on the login/signup page.
    """
    message = str(exc)
    if "already registered" in message.lower():
        return "An account with that email already exists. Try logging in instead."
    if "invalid login credentials" in message.lower():
        return "Invalid email or password."
    if "password" in message.lower() and "short" in message.lower():
        return "Password is too short. Use at least 6 characters."
    return "Something went wrong. Please try again."