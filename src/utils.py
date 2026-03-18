"""prana — utility functions."""
import time, hashlib, logging, functools
from typing import Any, Callable, TypeVar, Optional

logger = logging.getLogger(__name__)
T = TypeVar("T")

def generate_id() -> str:
    """Generate a short unique ID."""
    return hashlib.md5(f"{time.time()}{id(object())}".encode()).hexdigest()[:12]

def retry(max_retries: int = 3, delay: float = 0.1):
    """Decorator for retrying failed operations with exponential backoff."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_error = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        wait = delay * (2 ** attempt)
                        logger.warning(f"Retry {attempt + 1}/{max_retries} for {func.__name__}: {e}")
                        time.sleep(wait)
            raise last_error
        return wrapper
    return decorator

def sanitize_input(text: Any, max_length: int = 10000) -> str:
    """Sanitize and truncate input."""
    if isinstance(text, str):
        return text.strip()[:max_length]
    if text is None:
        return ""
    return str(text)[:max_length]

def timed(func: Callable) -> Callable:
    """Decorator that logs execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = (time.time() - start) * 1000
        logger.debug(f"{func.__name__} completed in {elapsed:.1f}ms")
        return result
    return wrapper

class SimpleCache:
    """Simple in-memory cache with TTL."""

    def __init__(self, ttl: float = 60.0):
        self.ttl = ttl
        self._store: dict = {}
        self._expiry: dict = {}

    def get(self, key: str) -> Optional[Any]:
        if key in self._store:
            if time.time() < self._expiry.get(key, 0):
                return self._store[key]
            del self._store[key]
            del self._expiry[key]
        return None

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        self._store[key] = value
        self._expiry[key] = time.time() + (ttl or self.ttl)

    def clear(self) -> None:
        self._store.clear()
        self._expiry.clear()

    @property
    def size(self) -> int:
        return len(self._store)
