"""Live timing: feed → state → decisions, during a session.

The offline pipeline fits the tyre model on practice long runs and seals it.
This package is what runs *on the day*: it consumes the official F1 live timing
feed (or a recording of one), keeps a tidy, lap-level view of the session in
memory, and turns the sealed model into decisions that change lap by lap —
when the tyre is done, when the pit window opens, whether the car behind can
undercut, what a safety car is worth.

Every source produces the same `Message` stream and every consumer reads the
same `LiveState`, so a recorded session replays through exactly the code path
the live feed uses.  That is what makes the live path testable before a session
exists.
"""

from src.live.streams import Message  # noqa: F401
from src.live.state import LiveState  # noqa: F401
