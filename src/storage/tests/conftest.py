"""Shared fixtures for storage tests."""


def _debug_import_snapshot():
    """Debug helper (not a test): check what sqlite_vec resolves to during pytest.

    Call manually from the console, not via pytest collection.
    """
    import sys
    print("\n=== DEBUG ===")
    print("sys.modules keys with 'sqlite':", [k for k in sys.modules if 'sqlite' in k])
    if 'sqlite_vec' in sys.modules:
        mod = sys.modules['sqlite_vec']
        print(f"sqlite_vec module: {mod.__file__}")
        print(f"Has load: {hasattr(mod, 'load')}")
    if 'storage.sqlite_vec' in sys.modules:
        mod = sys.modules['storage.sqlite_vec']
        print(f"storage.sqlite_vec._sqlite_vec: {getattr(mod, '_sqlite_vec', 'NOT SET')}")
    print("=== END DEBUG ===\n")
