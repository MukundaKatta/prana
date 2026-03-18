"""Tests for utility functions."""
import time
from src.utils import generate_id, sanitize_input, SimpleCache, retry

def test_generate_id():
    id1 = generate_id()
    id2 = generate_id()
    assert len(id1) == 12
    assert id1 != id2

def test_sanitize_input_string():
    assert sanitize_input("  hello  ") == "hello"

def test_sanitize_input_none():
    assert sanitize_input(None) == ""

def test_sanitize_input_truncate():
    long = "x" * 20000
    assert len(sanitize_input(long)) == 10000

def test_cache_set_get():
    cache = SimpleCache(ttl=10)
    cache.set("key", "value")
    assert cache.get("key") == "value"

def test_cache_miss():
    cache = SimpleCache()
    assert cache.get("missing") is None

def test_cache_expiry():
    cache = SimpleCache(ttl=0.01)
    cache.set("key", "value")
    time.sleep(0.02)
    assert cache.get("key") is None

def test_cache_clear():
    cache = SimpleCache()
    cache.set("a", 1)
    cache.set("b", 2)
    assert cache.size == 2
    cache.clear()
    assert cache.size == 0

def test_retry_success():
    call_count = 0
    @retry(max_retries=3, delay=0.01)
    def flaky():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ValueError("fail")
        return "ok"
    assert flaky() == "ok"
    assert call_count == 2
