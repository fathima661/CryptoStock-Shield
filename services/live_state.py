import time
from collections import deque

LIVE_STATE = {}
TTL = 120  # keep recent live sequence alive for 2 mins


def get_state(symbol):
    state = LIVE_STATE.get(symbol)
    if not state:
        return None

    if time.time() - state["ts"] > TTL:
        del LIVE_STATE[symbol]
        return None

    return state["data"]


def append_state(symbol, point, maxlen=20):
    existing = get_state(symbol)

    if existing is None:
        existing = deque(maxlen=maxlen)
    elif not isinstance(existing, deque):
        existing = deque(existing, maxlen=maxlen)

    existing.append(point)

    LIVE_STATE[symbol] = {
        "data": existing,
        "ts": time.time()
    }

    return list(existing)