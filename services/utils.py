# services/utils.py

import time
import logging

logger = logging.getLogger(__name__)


# ==========================================
# 🔁 RETRY DECORATOR
# ==========================================
def retry(max_attempts=3, delay=1):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if "RATE_LIMIT" in str(e):
                        logger.error("Rate limit hit — stopping retries")
                        raise e
                    logger.warning(f"{func.__name__} failed (attempt {attempt+1}): {str(e)}")
                    time.sleep(delay)

            raise Exception(f"{func.__name__} failed after {max_attempts} attempts")

        return wrapper
    return decorator


# ==========================================
# 🛡️ SAFE EXECUTION
# ==========================================
def safe_execute(func, fallback=None):
    try:
        return func()
    except Exception as e:
        logger.error(f"Safe execution failed: {str(e)}")
        return fallback