"""What "SOFT" meant that year — Pirelli's nomination, race by race.

The history priors pool what the field *did* at a circuit: start compounds,
sequences, first-stop laps.  All of it is recorded in roles — SOFT, MEDIUM,
HARD — and a role is not a tyre.  Pirelli nominates three C-numbers per event
from a seven-step range, and the choice moves between seasons: Barcelona ran
C1/C2/C3 in 2023 and 2025 and runs **C2/C3/C4** in 2026, so the 2025 race's
"SOFT" (C3) is this year's MEDIUM.  Pooled on the letters, the prior said the
Barcelona field starts on the SOFT 31 times out of 37 — and then asked the
2026 optimiser to start on a compound one step softer than anything those
races used.  Melbourne 2023 is the same trap one step the other way: C2/C3/C4
against 2026's C3/C4/C5, so its nine M-H one-stoppers are both C-numbers the
2026 field calls HARD, a plan that does not exist this year.

So every historical plan is translated through the C-numbers before it is
counted: role -> C-number under *that* year's nomination -> role under the
target year's.  The translation is exact where the two nominations share the
C-number and **clamped** where they do not (a compound harder than the
target's hardest becomes its HARD), and a sequence whose stints collapse onto
one compound under the mapping is not a legal plan at the target at all — the
caller drops it from the sequence counts, which is what `history.plan_prior_for`
does.

The table is `data/pirelli_nominations.json`, hard -> soft per event,
keyed by the circuit name `src.config` uses.  Values the planner could not
verify on press.pirelli.com / formula1.com are **absent, never guessed**
(2026 Silverstone, the 2026 rounds after Madring): an absent nomination makes
the comparison "unknown" and the caller falls back rather than inventing a
mapping.
"""

from __future__ import annotations

import json
from functools import lru_cache

from src.config import ROOT

# Not under `data/raw/`: that directory is gitignored (it holds the FastF1
# cache), and this table is hand-verified source data the repository must carry.
NOMINATIONS_FILE = ROOT / "data" / "pirelli_nominations.json"

# The order the table lists each event's three dry compounds, and the role each
# slot plays: index 0 is the hardest tyre of the weekend.
ROLE_ORDER = ("HARD", "MEDIUM", "SOFT")
LETTER = {"HARD": "H", "MEDIUM": "M", "SOFT": "S"}
_BY_LETTER = {"H": "HARD", "M": "MEDIUM", "S": "SOFT"}


# --------------------------------------------------------------------------
# The table
# --------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _table() -> dict:
    """`{year: {normalised circuit name: [hard, medium, soft]}}`.

    Every key the calendar, FastF1 or a human might use for a circuit is
    indexed: the calendar name normalised, plus `history.CIRCUIT_ALIASES` for
    the circuits whose common name shares no word with it (Monaco for Monte
    Carlo, Sao Paulo for Interlagos).
    """
    from src.history import CIRCUIT_ALIASES, _norm

    if not NOMINATIONS_FILE.exists():
        return {}
    raw = json.loads(NOMINATIONS_FILE.read_text())
    out: dict = {}
    for year, per_circuit in raw.items():
        if year.startswith("_") or not isinstance(per_circuit, dict):
            continue
        idx: dict = {}
        for name, nom in per_circuit.items():
            # CIRCUIT_ALIASES is keyed by the lower-cased calendar name, which
            # for "Yas Marina Circuit" is not what `_norm` leaves ("circuit" is
            # one of the words it drops), so both spellings are tried.
            aliases = CIRCUIT_ALIASES.get(_norm(name), []) or CIRCUIT_ALIASES.get(name.lower(), [])
            keys = {_norm(name)} | {_norm(a) for a in aliases}
            for k in keys:
                idx[k] = [str(c).upper() for c in nom]
        out[str(year)] = idx
    return out


def nomination(year: int | str, circuit: str) -> list[str] | None:
    """The dry nomination at `circuit` in `year`, hardest first, or None.

    None means "not verified", never "no nomination": the caller must fall
    back rather than assume the C-numbers of a neighbouring year.
    """
    from src.history import _norm

    per_year = _table().get(str(year))
    if not per_year:
        return None
    nom = per_year.get(_norm(circuit))
    return list(nom) if nom else None


def sources() -> list:
    """Where the table's values were verified; carried into the report."""
    if not NOMINATIONS_FILE.exists():
        return []
    return list(json.loads(NOMINATIONS_FILE.read_text()).get("_sources") or [])


# --------------------------------------------------------------------------
# Roles, C-numbers and the mapping between two nominations
# --------------------------------------------------------------------------


def _ok(nom) -> bool:
    return bool(nom) and len(nom) == len(ROLE_ORDER)


def _num(cnum) -> float | None:
    """The step of a C-number: "C3" -> 3.0.  None if it is not one."""
    s = str(cnum or "").strip().upper().lstrip("C")
    try:
        return float(s)
    except ValueError:
        return None


def role_of(compound) -> str | None:
    """"S", "soft", "SOFT" -> "SOFT"; an intermediate or an unknown -> None."""
    s = str(compound or "").strip().upper()
    if s in ROLE_ORDER:
        return s
    return _BY_LETTER.get(s[:1]) if s else None


def compound_cnumber(letter_or_name, nom) -> str | None:
    """Which C-number a role was, under `nom`.  None if either is unknown."""
    role = role_of(letter_or_name)
    if role is None or not _ok(nom):
        return None
    return str(nom[ROLE_ORDER.index(role)])


def cnumber_letter(cnum, nom) -> tuple[str | None, bool]:
    """The role a C-number plays under `nom`, as a letter, and a clamp flag.

    Exact where `nom` contains the C-number.  Otherwise the nearest step in
    `nom` (ties to the harder one), which covers both ends — a tyre harder
    than the target's hardest is its HARD, softer than its softest is its SOFT
    — and the interior gap a non-consecutive nomination leaves (2025 Spa ran
    C1/C3/C4, so a C2 has no exact role there).  The flag says the answer is
    an approximation, so the caller can report how much of the prior was
    clamped rather than mapped.
    """
    if not _ok(nom):
        return None, False
    want = _num(cnum)
    steps = [_num(c) for c in nom]
    if want is None or any(s is None for s in steps):
        return None, False
    for i, s in enumerate(steps):
        if s == want:
            return LETTER[ROLE_ORDER[i]], False
    # nearest, ties to the harder end (the hard end is index 0, lowest C-number)
    i = min(range(len(steps)), key=lambda j: (abs(steps[j] - want), steps[j]))
    return LETTER[ROLE_ORDER[i]], True


def _shared(from_nom, to_nom) -> int:
    if not _ok(from_nom) or not _ok(to_nom):
        return 0
    return len({_num(c) for c in from_nom} & {_num(c) for c in to_nom})


def _shift(from_nom, to_nom) -> int:
    """How many ladder steps the same rubber moves in role, from -> to.

    The median, over the C-numbers both nominations contain, of (role index
    under `from_nom` - role index under `to_nom`) with 0 = HARD.  +1 means the
    target's range is one step softer, so a compound that was the MEDIUM is
    the target's HARD (Barcelona 2023/2025 -> 2026).  0 means the roles line
    up even when the nominations differ (2025 Spa's C1/C3/C4 against 2026's
    C2/C3/C4: the medium and the soft are the same tyres).
    """
    if not _ok(from_nom) or not _ok(to_nom):
        return 0
    deltas = []
    for i, c in enumerate(from_nom):
        for j, d in enumerate(to_nom):
            if _num(c) is not None and _num(c) == _num(d):
                deltas.append(i - j)
    if not deltas:
        return 0
    deltas.sort()
    return int(deltas[len(deltas) // 2])


def comparable(from_nom, to_nom) -> str:
    """How far the two nominations can be compared.

        identical   the same three C-numbers: the letters mean the same thing
        shifted     at least two C-numbers in common: every stint can be
                    mapped, some roles move, the odd one clamps
        disjoint    one C-number or none in common: the history's letters say
                    nothing about the target's compounds
        unknown     one of the two nominations is not in the table
    """
    if not _ok(from_nom) or not _ok(to_nom):
        return "unknown"
    if [str(c).upper() for c in from_nom] == [str(c).upper() for c in to_nom]:
        return "identical"
    return "shifted" if _shared(from_nom, to_nom) >= 2 else "disjoint"


def map_sequence(seq, from_nom, to_nom) -> tuple[list, dict]:
    """Translate a compound sequence from one year's roles into another's.

    `seq` is letters ("S", "M", "H") or names ("SOFT", ...), in stint order;
    the result is always **letters**, so "-".join(...) is the short form the
    plan prior and the strategy search key on.  Returns
    `(letters, {"clamped", "shift", "shared"})`; `clamped` is True if any
    stint's C-number has no exact role at the target.

    With either nomination unknown the roles are returned unchanged — the
    honest degenerate case, which `comparable` reports as "unknown" so the
    caller can decide not to use it.
    """
    letters, clamped = [], False
    usable = _ok(from_nom) and _ok(to_nom)
    for item in seq:
        role = role_of(item)
        fallback = LETTER.get(role) or str(item).strip().upper()[:1]
        if not usable:
            letters.append(fallback)
            continue
        lt, cl = cnumber_letter(compound_cnumber(item, from_nom), to_nom)
        letters.append(lt or fallback)
        clamped = clamped or cl or lt is None
    return letters, {"clamped": bool(clamped), "shift": _shift(from_nom, to_nom),
                     "shared": _shared(from_nom, to_nom)}
