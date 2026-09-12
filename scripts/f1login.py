"""Store an F1TV subscription token for the live feed.

    make login                       opens a browser window: sign in, done
    make login PASTE=1               paste the formula1.com login-session cookie instead

The browser route runs its own Chromium (Playwright) with a persistent profile
under data/raw/browser, so a returning user is usually still signed in.  It
waits for the `login-session` cookie that formula1.com sets after sign-in,
takes the subscription token out of it, verifies the token against Formula 1's
public keys, and stores it where FastF1 and the live daemon read it.  No
extension, nothing to copy.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LOGIN_URL = "https://account.formula1.com/#/en/login"
COOKIE = "login-session"
PROFILE_DIR = ROOT / "data" / "raw" / "browser"


def extract_token(text: str) -> str | None:
    t = text.strip().strip('"').strip("'")
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


def store(tok: str) -> int:
    from fastf1.internals.f1auth import AUTH_DATA_FILE, JWKS_URL, _verify_jwt

    try:
        payload = _verify_jwt(tok, JWKS_URL)
    except Exception as exc:
        print(f"The token did not verify against Formula 1's keys: {exc}")
        return 1
    exp = payload.get("exp")
    when = datetime.fromtimestamp(exp, timezone.utc).strftime("%Y-%m-%d %H:%M UTC") if exp else "unknown"
    Path(AUTH_DATA_FILE).write_text(tok)
    print(f"TOKEN_STORED subscription={payload.get('SubscriptionStatus')}/{payload.get('SubscribedProduct')} "
          f"expires={when}", flush=True)
    return 0


def via_paste() -> int:
    text = input("Paste the login-session cookie value (or the token): ")
    tok = extract_token(text)
    if not tok:
        print("Could not find a subscriptionToken in what was pasted.")
        return 1
    return store(tok)


def via_browser(timeout_s: int = 900) -> int:
    from playwright.sync_api import sync_playwright

    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        ctx = pw.chromium.launch_persistent_context(
            str(PROFILE_DIR), headless=False, viewport={"width": 1100, "height": 850},
            args=["--disable-blink-features=AutomationControlled"])
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        # already signed in from a previous run?
        tok = _find_token(ctx)
        if tok:
            print("Already signed in from a previous run.", flush=True)
            ctx.close()
            return store(tok)
        page.goto(LOGIN_URL)
        print("BROWSER_OPEN sign in to your F1 account in the window that just opened; "
              "this closes by itself afterwards.", flush=True)
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            time.sleep(2)
            try:
                tok = _find_token(ctx)
            except Exception:
                tok = None
            if tok:
                rc = store(tok)
                time.sleep(1)
                ctx.close()
                return rc
            if not ctx.pages:
                print("Browser window closed before sign-in completed.", flush=True)
                return 1
        print("Timed out waiting for sign-in.", flush=True)
        ctx.close()
        return 1


def _find_token(ctx) -> str | None:
    for c in ctx.cookies():
        if c.get("name") == COOKIE and "formula1.com" in c.get("domain", ""):
            tok = extract_token(c.get("value", ""))
            if tok:
                return tok
    return None


def main() -> int:
    if "--paste" in sys.argv:
        return via_paste()
    try:
        return via_browser()
    except ImportError:
        print("Playwright is not installed; falling back to paste mode (pip install playwright; "
              "python -m playwright install chromium).")
        return via_paste()


if __name__ == "__main__":
    raise SystemExit(main())
