"""
Thin wrapper around the Supabase Python client for auth + history.
"""
from __future__ import annotations
import os
from datetime import datetime, timezone
from typing import Any, Dict, List
from supabase import create_client, Client

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    raise RuntimeError(
        "SUPABASE_URL and SUPABASE_ANON_KEY must be set in your .env file."
    )

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

_admin_client: Client | None = None


class AuthError(Exception):
    pass


class AdminError(Exception):
    pass


def _extract_username(user_obj) -> str | None:
    """Pulls the username out of Supabase user_metadata, or returns None."""
    try:
        meta = getattr(user_obj, "user_metadata", None) or {}
        return meta.get("username") or None
    except Exception:
        return None


def sign_up(email: str, password: str) -> Dict[str, Any]:
    try:
        result = supabase.auth.sign_up({"email": email, "password": password})
    except Exception as exc:
        raise AuthError(_clean_error(exc)) from exc

    if result.user is None:
        raise AuthError("Sign up failed. Please try again.")

    if result.session is None:
        raise AuthError(
            "Account created but email confirmation is required. "
            "Please contact support."
        )

    return {
        "id": result.user.id,
        "email": result.user.email,
        "access_token": result.session.access_token,
        "username": _extract_username(result.user),
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
    return {
        "id": result.user.id,
        "email": result.user.email,
        "access_token": access_token,
        "username": _extract_username(result.user),
    }


def sign_out() -> None:
    try:
        supabase.auth.sign_out()
    except Exception:
        pass


def send_password_reset(email: str, redirect_url: str | None = None) -> None:
    try:
        options = {}
        if redirect_url:
            options["redirect_to"] = redirect_url
        supabase.auth.reset_password_email(email, options)
    except Exception as exc:
        raise AuthError(_clean_error(exc)) from exc


def update_password(access_token: str, refresh_token: str | None, new_password: str) -> None:
    try:
        client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
        client.auth.set_session(access_token, refresh_token or "")
        client.auth.update_user({"password": new_password})
    except Exception as exc:
        raise AuthError(
            "Could not update your password. Your reset link may have expired — "
            "please request a new one."
        ) from exc


def update_username(access_token: str, username: str) -> str:
    """
    Saves a display username into the user's Supabase metadata.
    Returns the cleaned username on success.
    Raises AuthError on failure.
    """
    username = username.strip()
    if not username:
        raise AuthError("Username cannot be empty.")
    if len(username) > 32:
        raise AuthError("Username must be 32 characters or fewer.")

    try:
        client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
        client.auth.set_session(access_token, "")
        client.auth.update_user({"data": {"username": username}})
    except Exception as exc:
        raise AuthError("Could not update username. Please try again.") from exc

    return username


def delete_own_account(user_id: str) -> None:
    """
    Permanently deletes the calling user's own account.
    Uses the service_role client since GoTrue doesn't allow self-deletion
    via the anon key. The caller (app.py) must verify the user is
    authenticated before calling this.
    """
    client = _admin()
    try:
        client.auth.admin.delete_user(user_id)
    except Exception as exc:
        raise AuthError("Could not delete your account. Please try again.") from exc


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
    if "banned" in msg or "user is banned" in msg:
        return "This account has been restricted."
    if "rate limit" in msg:
        return "Too many attempts. Please wait a moment and try again."
    return "Something went wrong. Please try again."


# ---------------------------------------------------------------------------
# Admin operations (require SUPABASE_SERVICE_ROLE_KEY)
# ---------------------------------------------------------------------------

def _admin() -> Client:
    global _admin_client
    if _admin_client is None:
        if not SUPABASE_SERVICE_ROLE_KEY:
            raise AdminError(
                "SUPABASE_SERVICE_ROLE_KEY is not set. "
                "Add it in Render's environment variables to enable the admin panel."
            )
        _admin_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    return _admin_client


def list_users(per_page: int = 1000) -> List[Dict[str, Any]]:
    client = _admin()
    try:
        response = client.auth.admin.list_users(page=1, per_page=per_page)
    except Exception as exc:
        raise AdminError(f"Could not load users: {exc}") from exc

    users = []
    for u in response:
        banned_until = getattr(u, "banned_until", None)
        is_banned = bool(banned_until) and _is_future(banned_until)
        meta = getattr(u, "user_metadata", None) or {}
        users.append({
            "id": u.id,
            "email": u.email,
            "username": meta.get("username") or None,
            "created_at": str(getattr(u, "created_at", "")),
            "banned": is_banned,
        })
    users.sort(key=lambda u: u["created_at"], reverse=True)
    return users


def _is_future(value: Any) -> bool:
    try:
        if isinstance(value, str):
            ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
        else:
            ts = value
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts > datetime.now(timezone.utc)
    except Exception:
        return True


def ban_user(user_id: str) -> None:
    client = _admin()
    try:
        client.auth.admin.update_user_by_id(user_id, {"ban_duration": "876000h"})
    except Exception as exc:
        raise AdminError(f"Could not ban user: {exc}") from exc


def unban_user(user_id: str) -> None:
    client = _admin()
    try:
        client.auth.admin.update_user_by_id(user_id, {"ban_duration": "none"})
    except Exception as exc:
        raise AdminError(f"Could not unban user: {exc}") from exc


def delete_user(user_id: str) -> None:
    client = _admin()
    try:
        client.auth.admin.delete_user(user_id)
    except Exception as exc:
        raise AdminError(f"Could not delete user: {exc}") from exc


def total_prediction_count() -> int:
    client = _admin()
    try:
        response = client.table("predictions").select("id", count="exact").execute()
    except Exception as exc:
        raise AdminError(f"Could not count predictions: {exc}") from exc
    return response.count or 0