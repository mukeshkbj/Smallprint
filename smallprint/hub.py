"""Small helpers for publishing to the Hugging Face Hub."""

from __future__ import annotations

import time
import traceback

# The Hub only accepts licence ids from its list (https://huggingface.co/docs/hub/repositories-licenses).
# CC BY-NC 2.5 (the multilingual CLAUDETTE licence) isn't on it, so it is declared as "other".
CC_BY_NC_25 = (
    "license: other\n"
    "license_name: cc-by-nc-2.5\n"
    "license_link: https://creativecommons.org/licenses/by-nc/2.5/\n"
)


def safe_push(what: str, fn, *args, retries: int = 3, wait: float = 10.0, **kwargs) -> bool:
    """Run an upload without letting a Hub error end a long pipeline. The local files stay
    in place, so a failed upload can be redone later. Returns True on success."""
    for attempt in range(1, retries + 1):
        try:
            fn(*args, **kwargs)
            return True
        except Exception as e:  # noqa: BLE001 - any Hub/network error
            msg = str(e).strip().splitlines()[-1] if str(e).strip() else type(e).__name__
            print(f"[hub] {what}: upload failed (attempt {attempt}/{retries}): {msg}", flush=True)
            if attempt == retries or "Bad request" in str(e) or "must be" in str(e):
                traceback.print_exc()
                print(f"[hub] {what}: giving up; files are still on disk", flush=True)
                return False
            time.sleep(wait)
    return False
