"""Keeping the F1TV subscription token fresh, without making it a chore.

F1TV subscription tokens last about four days and Formula 1 publishes no
refresh endpoint, so a token that worked on Friday is dead by Tuesday — and it
dies quietly, because every *timing* topic works without it.  Only the car
telemetry (`CarData.z` / `Position.z`) needs one.

The sign-in itself, though, is sticky.  `make login` drives a Chromium profile
kept under `data/raw/browser`, and formula1.com leaves its `login-session`
cookie in that profile for far longer than the token lasts.  So the common
case — "my token expired, I am still signed in" — can be repaired with nobody
touching anything: open the same profile headless, read the cookie, mint a new
token.  Only when the cookie has lapsed too does anyone have to sign in again.

Nothing in here may ever stop the feed from starting.  Every entry point
returns a state and a sentence to print; none of them raise, and none of them
block for longer than they were told to.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

from src.live.sources import _token_expiry, f1tv_token_status

log = logging.getLogger("degless.live.auth")

ROOT = Path(__file__).resolve().parents[2]
PROFILE_DIR = ROOT / "data" / "raw" / "browser"
LOGIN_URL = "https://account.formula1.com/#/en/login"
COOKIE = "login-session"

# Refresh this long before the token actually dies, so a session never starts
# on a token that will expire halfway through it.
RENEW_WITHIN_H = 12.0


# --------------------------------------------------------------------------
# The token itself
# --------------------------------------------------------------------------


def extract_token(text: str) -> str | None:
    """The subscription token out of a raw cookie value, JSON blob, or JWT."""
    t = (text or "").strip().strip('"').strip("'")
    if t.startswith("eyJ"):
        return t
    decoded = urllib.parse.unquote(t)
    for cand in (decoded, t):
        try:
            d = json.loads(cand)
        except Exception:
            continue
        tok = (d.get("data") or {}).get("subscriptionToken") or d.get("subscriptionToken")
        if tok:
            return tok
    return None


def store(tok: str) -> tuple[bool, str]:
    """Verify a token against Formula 1's public keys and save it for FastF1."""
    try:
        from fastf1.internals.f1auth import AUTH_DATA_FILE, JWKS_URL, _verify_jwt
    except Exception as exc:
        return False, f"FastF1 auth support unavailable ({exc})"
    try:
        payload = _verify_jwt(tok, JWKS_URL)
    except Exception as exc:
        return False, f"the token did not verify against Formula 1's keys: {exc}"
    exp = payload.get("exp")
    when = (datetime.fromtimestamp(exp, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            if exp else "unknown")
    try:
        Path(AUTH_DATA_FILE).write_text(tok)
    except Exception as exc:
        return False, f"could not save the token ({exc})"
    return True, (f"F1TV token stored — subscription "
                  f"{payload.get('SubscriptionStatus')}/{payload.get('SubscribedProduct')}, "
                  f"expires {when}")


def hours_left(tok: str | None = None) -> float | None:
    """Hours until the stored (or given) token expires; negative once it has."""
    if tok is None:
        tok, _, _ = f1tv_token_status()
        if tok is None:
            return None
    exp = _token_expiry(tok)
    if exp is None:
        return None
    return (exp - datetime.now(timezone.utc)).total_seconds() / 3600.0


def status() -> tuple[str | None, str, str]:
    """`(token, status, detail)` — `ok` | `expired` | `invalid` | `none`."""
    return f1tv_token_status()


def needs_refresh() -> tuple[bool, str, str]:
    """`(should_refresh, status, detail)`, counting "about to expire" as due."""
    tok, st, detail = f1tv_token_status()
    if tok is None:
        return True, st, detail
    left = hours_left(tok)
    if left is not None and left < RENEW_WITHIN_H:
        return True, "expiring", f"F1TV token expires in {left:.1f} h"
    return False, "ok", detail if left is None else f"F1TV token valid for {left/24:.1f} days"


# --------------------------------------------------------------------------
# Getting a new one
# --------------------------------------------------------------------------


def _token_from_context(ctx) -> str | None:
    for c in ctx.cookies():
        if c.get("name") == COOKIE and "formula1.com" in c.get("domain", ""):
            tok = extract_token(c.get("value", ""))
            if tok:
                return tok
    return None


def silent_refresh(timeout_s: float = 60.0) -> tuple[bool, str]:
    """Mint a new token from the signed-in browser profile, without a window.

    This is the path that makes an expired token a non-event: the profile
    under `data/raw/browser` usually still holds a valid `login-session`
    cookie, which is all a new token needs.
    """
    if not PROFILE_DIR.exists():
        return False, "no saved browser profile yet; a first sign-in is needed"
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return False, "playwright is not installed; cannot refresh without a browser"
    ctx = None
    try:
        with sync_playwright() as pw:
            ctx = pw.chromium.launch_persistent_context(
                str(PROFILE_DIR), headless=True, timeout=timeout_s * 1000,
                args=["--disable-blink-features=AutomationControlled"])
            tok = _token_from_context(ctx)
            if tok is None:
                # The cookie can need a page load to be re-issued.
                page = ctx.pages[0] if ctx.pages else ctx.new_page()
                try:
                    page.goto("https://www.formula1.com/", timeout=timeout_s * 1000,
                              wait_until="domcontentloaded")
                except Exception:
                    pass
                deadline = time.time() + min(timeout_s, 20.0)
                while tok is None and time.time() < deadline:
                    time.sleep(1.0)
                    tok = _token_from_context(ctx)
            ctx.close()
            ctx = None
            if tok is None:
                return False, "the saved browser profile is no longer signed in"
            return store(tok)
    except Exception as exc:
        return False, f"silent refresh failed ({exc})"
    finally:
        if ctx is not None:
            try:
                ctx.close()
            except Exception:
                pass


def browser_login(timeout_s: float = 900.0) -> tuple[bool, str]:
    """Open the sign-in window and wait for the cookie to appear."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return False, ("playwright is not installed (pip install playwright && "
                       "python -m playwright install chromium); use paste mode instead")
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with sync_playwright() as pw:
            ctx = pw.chromium.launch_persistent_context(
                str(PROFILE_DIR), headless=False, viewport={"width": 1100, "height": 850},
                args=["--disable-blink-features=AutomationControlled"])
            page = ctx.pages[0] if ctx.pages else ctx.new_page()
            tok = _token_from_context(ctx)
            if tok:
                ctx.close()
                return store(tok)
            page.goto(LOGIN_URL)
            print("  a browser window has opened — sign in to your F1 account; "
                  "it closes by itself afterwards.", flush=True)
            t0 = time.time()
            while time.time() - t0 < timeout_s:
                time.sleep(2)
                try:
                    tok = _token_from_context(ctx)
                except Exception:
                    tok = None
                if tok:
                    ok, msg = store(tok)
                    time.sleep(1)
                    ctx.close()
                    return ok, msg
                if not ctx.pages:
                    return False, "the browser window was closed before sign-in finished"
            ctx.close()
            return False, "timed out waiting for sign-in"
    except Exception as exc:
        return False, f"browser sign-in failed ({exc})"


def paste_login(text: str | None = None) -> tuple[bool, str]:
    """Take the `login-session` cookie value (or a raw token) as pasted text."""
    if text is None:
        try:
            text = input("Paste the login-session cookie value (or the token): ")
        except (EOFError, KeyboardInterrupt):
            return False, "nothing pasted"
    tok = extract_token(text)
    if not tok:
        return False, "could not find a subscriptionToken in what was pasted"
    return store(tok)


def refresh(*, allow_browser: bool = True, timeout_s: float = 900.0) -> tuple[bool, str]:
    """Silent refresh first; fall back to the sign-in window."""
    ok, msg = silent_refresh()
    if ok:
        return True, msg
    log.debug("silent refresh: %s", msg)
    if not allow_browser:
        return False, msg
    return browser_login(timeout_s=timeout_s)
