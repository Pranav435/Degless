"""Palette, chart chrome and the few components every tab is built from.

TGR Haas red, black and white, in two modes.  Dark mode: a black canvas,
white text, red as the accent (selected controls, the tab underline, primary
buttons, the number tiles) and a red sidebar.  Light mode swaps black for
white: a white canvas, black text, the same red accent and sidebar.  Tyre
colours are Pirelli's own on the dark canvas; the light canvas gets deeper
versions, because a white HARD or a yellow MEDIUM disappears on white.

Charts are written once, in the dark vocabulary below (WHITE is the ink, BLACK
the paper), and `style()` re-keys every colour to the active mode.  The page
is painted by app/theme.css from the --dg-* variables `inject_css()` writes.
"""

from __future__ import annotations

import html
import re
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

RED = "#E6002B"
RED_DARK = "#B30022"
BLACK = "#0E0E10"
BLACK_2 = "#1C1C21"
WHITE = "#FFFFFF"
MUTED = "rgba(255,255,255,0.62)"
DIM = "rgba(255,255,255,0.45)"
FAINT = "rgba(255,255,255,0.35)"
HAIR = "rgba(255,255,255,0.25)"
GRID = "rgba(255,255,255,0.10)"
FLAG = "#FFD12E"
UNKNOWN = "#8C8C91"

TYRE = {"SOFT": "#FF3B30", "MEDIUM": "#FFD12E", "HARD": "#F2F2F2"}
LADDER = ["SOFT", "MEDIUM", "HARD"]
LETTER = {"SOFT": "S", "MEDIUM": "M", "HARD": "H"}
_FROM_LETTER = {v: k for k, v in LETTER.items()}

# Everything that changes between the modes. "ink" and "paper" are what the
# chart code calls WHITE and BLACK; "css" is written to the page as --dg-*.
MODES = {
    "dark": {
        "ink": WHITE, "paper": BLACK, "paper2": BLACK_2, "unknown": UNKNOWN, "accent": "#FF2E45",
        "tyre": dict(TYRE),
        "css": {"page": BLACK, "card": "#17171B", "hair": "rgba(255,255,255,0.12)", "muted": MUTED,
                "tile": RED_DARK, "panel": "#17171B", "ink": WHITE, "accent": "#FF2E45",
                "pill": BLACK_2, "pill-edge": "rgba(255,255,255,0.28)"},
    },
    "light": {
        "ink": BLACK, "paper": WHITE, "paper2": "#E4E4E8", "unknown": "#6E6E76", "accent": RED,
        "tyre": {"SOFT": "#D9261C", "MEDIUM": "#B87A00", "HARD": "#7C7C84"},
        "css": {"page": WHITE, "card": "#F4F4F6", "hair": "rgba(14,14,16,0.12)", "muted": "rgba(14,14,16,0.62)",
                "tile": RED_DARK, "panel": "#ECECF0", "ink": BLACK, "accent": RED,
                "pill": BLACK, "pill-edge": BLACK},
    },
}

FONT = "'Source Sans Pro', 'Source Sans 3', system-ui, sans-serif"
_CSS = Path(__file__).with_name("theme.css")


def mode() -> str:
    """'dark' or 'light': whichever theme the viewer's browser is showing."""
    try:
        t = st.context.theme.type
    except Exception:
        t = None
    return t if t in MODES else "dark"


def palette() -> dict:
    return MODES[mode()]


def inject_css() -> None:
    css_vars = "".join(f"--dg-{k}:{v};" for k, v in palette()["css"].items())
    st.html(f"<style>:root{{{css_vars}}}\n{_CSS.read_text(encoding='utf-8')}</style>")


def _hex_rgb(h: str) -> tuple:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _colour_table(target: dict) -> tuple[dict, dict]:
    """Dark-vocabulary colour -> its value in `target`, as exact strings and as rgb triples."""
    dark = MODES["dark"]
    pairs = [(dark[k], target[k]) for k in ("ink", "paper", "paper2", "unknown")]
    pairs += [(dark["tyre"][c], target["tyre"][c]) for c in LADDER]
    exact = {d: t for d, t in pairs}
    triples = {_hex_rgb(d): _hex_rgb(t) for d, t in pairs}
    return exact, triples


_RGBA = re.compile(r"rgba\((\d+),(\d+),(\d+),([\d.]+)\)")


def _translate(value, exact: dict, triples: dict):
    """One colour string re-keyed to the target mode; anything else untouched."""
    if not isinstance(value, str):
        return value
    if value in exact:
        return exact[value]
    m = _RGBA.fullmatch(value)
    if m:
        rgb = tuple(int(m.group(i)) for i in (1, 2, 3))
        if rgb in triples:
            r, g, b = triples[rgb]
            return f"rgba({r},{g},{b},{m.group(4)})"
    return value


def _walk(obj, exact: dict, triples: dict):
    if isinstance(obj, dict):
        # The plotly template carries its own colours; leave it alone.
        return {k: (v if k == "template" else _walk(v, exact, triples)) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_walk(v, exact, triples) for v in obj]
    return _translate(obj, exact, triples)


def recolor(fig):
    """A figure written in the dark vocabulary, re-keyed to the active mode."""
    if mode() == "dark":
        return fig
    exact, triples = _colour_table(palette())
    d = fig.to_plotly_json()
    return go.Figure(data=_walk(d.get("data", []), exact, triples), layout=_walk(d.get("layout", {}), exact, triples))


# --------------------------------------------------------------------------
# Colour and number utilities
# --------------------------------------------------------------------------


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def ccol(c) -> str:
    return TYRE.get(str(c).upper(), UNKNOWN)


def in_ladder(compounds) -> list:
    """Compounds sorted softest-first, unknowns appended."""
    compounds = list(compounds)
    known = [c for c in LADDER if c in set(compounds)]
    return known + [c for c in compounds if c not in LADDER]


def ink_on(fill_hex: str, alpha: float = 1.0) -> str:
    """Dark or light label text for a fill blended onto the card, in the mode
    the chart will be shown in.  Returned in the dark vocabulary (BLACK is
    dark ink, WHITE light ink) so that `recolor()` lands it on the right side:
    in light mode WHITE becomes the black ink and BLACK the white paper."""
    pal = palette()
    exact, _ = _colour_table(pal)
    fill = _hex_rgb(exact.get(fill_hex, fill_hex))
    base = _hex_rgb(pal["css"]["card"])
    rgb = [alpha * f + (1 - alpha) * b for f, b in zip(fill, base)]
    lin = [(v / 255) / 12.92 if v / 255 <= 0.03928 else ((v / 255 + 0.055) / 1.055) ** 2.4 for v in rgb]
    lum = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    dark_ink = lum > 0.18
    if mode() == "dark":
        return BLACK if dark_ink else WHITE
    return WHITE if dark_ink else BLACK


def finite(x) -> bool:
    try:
        return x is not None and bool(np.isfinite(float(x)))
    except (TypeError, ValueError):
        return False


def fmt(x, nd: int = 1, suffix: str = "") -> str:
    return f"{float(x):.{nd}f}{suffix}" if finite(x) else "—"


def pct(x) -> str:
    return f"{float(x):.0%}" if finite(x) else "—"


def age(iso: str | None) -> str:
    if not iso:
        return "—"
    try:
        s = (datetime.now(timezone.utc) - datetime.fromisoformat(str(iso).replace("Z", "+00:00"))).total_seconds()
    except Exception:
        return "—"
    if s < 120:
        return f"{s:.0f} s ago"
    if s < 7200:
        return f"{s / 60:.0f} min ago"
    if s < 172800:
        return f"{s / 3600:.0f} h ago"
    return f"{s / 86400:.0f} days ago"


# --------------------------------------------------------------------------
# Words: plain labels built from structured fields
# --------------------------------------------------------------------------

DEFS = {
    "stint": "The laps run on one set of tyres.",
    "undercut": "Pitting before the car ahead so your fresh tyres gain enough time to come out in front.",
    "overcut": "Staying out while the car ahead pits, gaining time while they bring their new tyres up to speed.",
    "pit_window": "The laps where stopping costs less than 1 s compared with the best lap to stop.",
    "pit_loss": "Time lost driving through the pit lane and stopping, compared with staying on track.",
    "safety_car": "Under a safety car or VSC the field slows, so a stop costs much less time.",
    "drop_off": "The lap a tyre stops being usable and lap times fall away quickly.",
    "life_used": "How much of the tyre's usable life is gone. 100% is the drop-off.",
    "sims": "The model re-runs the race hundreds of times with slightly different tyre wear. "
            "Percentages are the share of those runs.",
    "likely_range": "Where the answer lands in 9 out of 10 simulated races.",
    "saving": "How much the driver looks after the tyres. Flat out is a practice long run; "
              "saving costs lap time but makes the tyre last longer.",
    "tyre_wear": "Seconds a tyre loses per lap as it ages, with fuel burn and the track getting faster taken out.",
    "race_vs_practice": "Tyres usually wear slower in a race than in practice long runs, because drivers manage them.",
    "wear_vs_forecast": "This car's tyre wear against the forecast. 1.20× means wearing 20% faster.",
}

_SAVING = {1.0: "Flat out", 0.85: "Light saving", 0.7: "Moderate saving", 0.55: "Heavy saving"}
_VERDICT = {"PIT": "Box", "STAY": "Stay out", "MARGINAL": "Close call", "PLANNED": "Planned stop"}
LEVEL_KIND = {"good": "ok", "accent": "info", "warn": "alert", "bad": "alert"}


def saving_word(push) -> str:
    if not finite(push):
        return "—"
    return _SAVING[min(_SAVING, key=lambda k: abs(k - float(push)))]


def verdict_word(code) -> str:
    return _VERDICT.get(str(code).upper(), str(code).title())


def stops_word(n) -> str:
    n = int(n)
    return "No stop" if n == 0 else f"{n}-stop"


def plan_text(compounds, pit_laps=None) -> str:
    """'1-stop M → H · lap 27' from a structured plan."""
    comps = [str(c).upper() for c in (compounds or [])]
    if not comps:
        return "—"
    s = f"{stops_word(len(comps) - 1)} " + " → ".join(LETTER.get(c, c[:1]) for c in comps)
    laps = [int(p) for p in (pit_laps or [])]
    if laps:
        s += (" · laps " if len(laps) > 1 else " · lap ") + ", ".join(str(p) for p in laps)
    return s


def event_text(key: str) -> str:
    """'hungary-2026' → 'Hungary 2026'."""
    return str(key).replace("-", " ").title()


def pit_loss_text(src) -> str:
    """Plain wording for a pit-loss source, including ones already saved in older files."""
    s = str(src or "")
    m = re.match(r"median of (\d+) donor races", s)
    if m:
        return f"average of {m.group(1)} other 2026 races"
    m = re.match(r"measured at ([\w-]+) \((\d+) stops\)", s)
    if m:
        return f"measured at {event_text(m.group(1))} ({m.group(2)} stops)"
    if s.startswith("default"):
        return "standard value; no race measured yet"
    m = re.match(r"this pit lane, \[?([\d, ]+)\]?", s)
    if m:
        return f"this pit lane, {m.group(1)}"
    return s


def window_line(windows: list, pits: list | None = None) -> str:
    """'Pit lap 23 — anywhere in 20–26 costs under 1 s'."""
    if windows:
        laps = " and ".join(str(w["recommended"]) for w in windows)
        rng = " / ".join(f"{w['lo']}–{w['hi']}" for w in windows)
        return f"Pit lap{'s' if len(windows) > 1 else ''} {laps} — anywhere in {rng} costs under 1 s"
    if pits:
        return "Pit lap" + ("s " if len(pits) > 1 else " ") + " and ".join(str(p) for p in pits)
    return "No stop"


_LABEL = re.compile(r"^(\d+)-stop ([SMH](?:-[SMH])*)(?: @ ([\d,]+))?$")


def label_text(label) -> str:
    """A stored plan label ('2-stop M-H-H @ 17,37') in the same form as plan_text."""
    m = _LABEL.match(str(label or "").strip())
    if not m:
        s = str(label or "—")
        return s[:1].upper() + s[1:]
    comps = [_FROM_LETTER[x] for x in m.group(2).split("-")]
    laps = [int(x) for x in m.group(3).split(",")] if m.group(3) else []
    return plan_text(comps, laps)


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------


def style(fig, height: int = 430, ytitle: str = "", xtitle: str = "", legend: bool = True):
    fig.update_layout(
        height=height, margin=dict(l=8, r=12, t=30, b=8),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=WHITE, size=13, family=FONT),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(color=MUTED)),
        hovermode="closest",
        hoverlabel=dict(bgcolor=BLACK, bordercolor=HAIR, font=dict(color=WHITE, family=FONT)),
        xaxis_title=xtitle, yaxis_title=ytitle,
    )
    for axis in (fig.update_xaxes, fig.update_yaxes):
        axis(gridcolor=GRID, zeroline=False, linecolor=HAIR, automargin=True,
             title_font=dict(color=MUTED), tickfont=dict(color=MUTED))
    return recolor(fig)


def chart(fig, key: str | None = None) -> None:
    st.plotly_chart(fig, width="stretch", theme=None, key=key, config={"displayModeBar": False})


# --------------------------------------------------------------------------
# Components
# --------------------------------------------------------------------------


def headline(text: str, sub: str = "", eyebrow: str = "") -> None:
    """The one big answer line a view leads with."""
    parts = [f"<div class='dg-eyebrow'>{eyebrow}</div>"] if eyebrow else []
    parts.append(f"<div class='dg-headline'>{text}</div>")
    if sub:
        parts.append(f"<div class='dg-sub'>{sub}</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def tiles(name: str, items) -> None:
    """A row of number tiles; each item is (label, value[, help[, sub]])."""
    items = [it for it in items if it]
    if not items:
        return
    with st.container(key=f"tiles-{name}", horizontal=True, gap="small"):
        for it in items:
            label, value = it[0], it[1]
            hlp = it[2] if len(it) > 2 else None
            sub = it[3] if len(it) > 3 else None
            st.metric(label, value, help=hlp, delta=sub or None, delta_color="off", delta_arrow="off")


@contextmanager
def card(name: str, title: str = "", tip: str | None = None, sub: str = ""):
    """A black card; the title states the answer, not the topic."""
    with st.container(key=f"card-{name}"):
        if title:
            st.subheader(title, anchor=False, help=tip)
        if sub:
            st.caption(sub)
        yield


_BADGE = {"ok": ("green", ":material/check_circle:"), "alert": ("red", ":material/warning:"),
          "info": ("gray", None), "flag": ("yellow", ":material/flag:")}


def badge(text: str, kind: str = "info", icon: str | None = None, help: str | None = None) -> None:
    color, default_icon = _BADGE.get(kind, _BADGE["info"])
    st.badge(text, color=color, icon=icon or default_icon, help=help)


def badges(items) -> None:
    """Several badges on one line; each item is (text, kind[, icon])."""
    parts = []
    for it in items:
        if not it:
            continue
        color, default_icon = _BADGE.get(it[1], _BADGE["info"])
        icon = it[2] if len(it) > 2 else default_icon
        text = str(it[0]).replace("[", "(").replace("]", ")")
        parts.append(f":{color}-badge[{icon + ' ' if icon else ''}{text}]")
    if parts:
        st.markdown(" ".join(parts))


def tyre_pill(c, text: str = "") -> str:
    """Inline HTML: black pill, tyre-coloured dot, white word."""
    c = str(c).upper()
    return (f"<span class='dg-tyre'><span class='dg-dot' style='background:{ccol(c)}'></span>"
            f"{html.escape(text or c.title())}</span>")


def how(*bullets: str) -> None:
    with st.expander("How this works", icon=":material/help:"):
        st.markdown("\n".join(f"- {b}" for b in bullets if b))


def more(label: str = "More detail"):
    return st.expander(label, icon=":material/add:")


def notice(text: str, kind: str = "info") -> None:
    """A short black note on the red page; text may carry <b>/<code> HTML."""
    icon = {"alert": "⚠ ", "ok": "✓ "}.get(kind, "")
    st.markdown(f"<div class='dg-notice dg-{kind}'>{icon}{text}</div>", unsafe_allow_html=True)
