"""Store an F1TV subscription token for the live feed.

    make login                       refresh quietly if possible, else open a browser
    make login PASTE=1               paste the formula1.com login-session cookie instead

`make run` does this by itself — it tops the token up on startup and every
half hour, and only asks for a sign-in when the saved browser profile can no
longer produce one.  This command is for doing it deliberately.

The browser route runs its own Chromium (Playwright) with a persistent profile
under data/raw/browser, so a returning user is usually still signed in and
nothing has to be typed.  It waits for the `login-session` cookie that
formula1.com sets after sign-in, takes the subscription token out of it,
verifies it against Formula 1's public keys, and stores it where FastF1 and
the live daemon read it.

The token unlocks car telemetry only; every timing topic works without one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.live import auth  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paste", action="store_true",
                    help="paste the login-session cookie instead of opening a browser")
    ap.add_argument("--status", action="store_true", help="report the stored token and exit")
    ap.add_argument("--quiet-only", action="store_true",
                    help="only try the silent refresh; never open a window")
    args = ap.parse_args()

    tok, state, detail = auth.status()
    print(f"stored token: {state} — {detail}")
    if args.status:
        return 0 if state == "ok" else 1

    if args.paste:
        ok, msg = auth.paste_login()
    else:
        ok, msg = auth.refresh(allow_browser=not args.quiet_only)
    print(msg)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
