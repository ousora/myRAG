"""Tests for resource cleanup and misc storage operations."""

import sqlite3

import pytest


class TestResourceCleanup:
    """Test connection lifecycle."""

    def test_close_closes_connection(self, store):
        store.close()
        with pytest.raises((sqlite3.OperationalError, sqlite3.ProgrammingError)):
            store.conn.execute("SELECT 1")
