"""Deep-merge of F1 live timing patches.

The live timing feed sends *patches*, not states: a message on `TimingData`
carries only the fields that changed.  Reconstructing the state is a recursive
merge with three rules the feed relies on:

* dict into dict merges key by key, recursing;
* a dict whose keys are all integer strings, arriving where the state holds a
  *list*, addresses list elements by index (`{"2": {"Value": "28.3"}}` patches
  the third sector) and may extend the list;
* a `_deleted` key carries a list of keys to remove from that level (used by
  `RaceControlMessages` and occasionally `DriverList`).

Everything else replaces.  The merge is in place and returns the merged object
so callers can write `state = merge(state, patch)` whether or not the base
existed.
"""

from __future__ import annotations

from typing import Any


def _is_index_dict(d: dict) -> bool:
    return bool(d) and all(isinstance(k, str) and k.isdigit() for k in d)


def merge(base: Any, patch: Any) -> Any:
    """Merge `patch` into `base` following the live timing conventions."""
    if isinstance(patch, dict):
        if isinstance(base, list):
            if _is_index_dict(patch):
                for k, v in patch.items():
                    i = int(k)
                    while len(base) <= i:
                        base.append({} if isinstance(v, dict) else None)
                    base[i] = merge(base[i], v)
                return base
            return patch if patch else base
        if not isinstance(base, dict):
            base = {}
        for k, v in patch.items():
            if k == "_deleted":
                if isinstance(v, (list, tuple)):
                    for dk in v:
                        base.pop(str(dk), None)
                continue
            if k in base and isinstance(v, (dict, list)):
                base[k] = merge(base[k], v)
            else:
                base[k] = v
        return base
    if isinstance(patch, list):
        # A list replaces a list wholesale; a list arriving over a dict is
        # what the feed does for an *initial* value (e.g. empty `Stints: []`).
        return patch
    return patch


def deep_copy(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: deep_copy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [deep_copy(v) for v in obj]
    return obj


def indexed(obj: Any) -> list:
    """A list-or-index-dict as an ordered list of (index, value)."""
    if isinstance(obj, list):
        return list(enumerate(obj))
    if isinstance(obj, dict):
        try:
            return sorted(((int(k), v) for k, v in obj.items()), key=lambda t: t[0])
        except ValueError:
            return []
    return []
