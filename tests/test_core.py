"""Tests for Prana."""
from src.core import Prana
def test_init(): assert Prana().get_stats()["ops"] == 0
def test_op(): c = Prana(); c.process(x=1); assert c.get_stats()["ops"] == 1
def test_multi(): c = Prana(); [c.process() for _ in range(5)]; assert c.get_stats()["ops"] == 5
def test_reset(): c = Prana(); c.process(); c.reset(); assert c.get_stats()["ops"] == 0
def test_service_name(): c = Prana(); r = c.process(); assert r["service"] == "prana"
