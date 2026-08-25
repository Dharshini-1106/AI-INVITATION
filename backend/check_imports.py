"""Quick dependency & import verification for the backend."""
import sys
import traceback


def main():
    results = []
    modules = [
        "fastapi", "uvicorn", "pydantic", "pydantic_settings",
        "scipy", "skimage", "PIL", "cv2", "numpy", "requests",
    ]
    ok = True
    for mod in modules:
        try:
            __import__(mod)
            results.append(f"OK   {mod}")
        except Exception as exc:  # noqa: BLE001
            ok = False
            results.append(f"FAIL {mod}: {exc!r}")

    # Try importing the full app (fallback mode should work)
    try:
        import app.main  # noqa: F401
        results.append("OK   app.main import")
    except Exception as exc:  # noqa: BLE001
        ok = False
        results.append(f"FAIL app.main import: {exc!r}")
        traceback.print_exc()

    print("\n".join(results))
    print("RESULT:", "ALL_OK" if ok else "HAS_FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

