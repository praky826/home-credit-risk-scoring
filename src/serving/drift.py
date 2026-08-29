from collections import deque
from typing import Dict, Any, List

# In-memory rolling window (thread-safe deque)
# maxlen defaults to 1000 as per SERVING_SPEC.md Startup Sequence step 5
_window = deque(maxlen=1000)

def add_request(features_dict: Dict[str, Any], probability: float) -> None:
    """Adds a request's input features and predicted probability to the rolling window."""
    record = features_dict.copy()
    record["probability"] = probability
    _window.append(record)

def get_window_records() -> List[Dict[str, Any]]:
    """Returns all records currently stored in the rolling window."""
    return list(_window)

def get_window_size() -> int:
    """Returns the number of records currently in the window."""
    return len(_window)

def clear_window() -> None:
    """Clears the rolling window."""
    _window.clear()
