"""Test the health endpoint and root endpoint of the running backend."""
import json
import urllib.request


def get(url):
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def main():
    status, body = get("http://127.0.0.1:8000/api/v1/health")
    print("HEALTH STATUS:", status)
    print("HEALTH BODY:", json.dumps(body) if isinstance(body, dict) else body)

    status2, body2 = get("http://127.0.0.1:8000/")
    print("ROOT STATUS:", status2)
    print("ROOT BODY:", json.dumps(body2) if isinstance(body2, dict) else body2)

    status3, body3 = get("http://127.0.0.1:8000/api/v1/pipeline/stages")
    print("STAGES STATUS:", status3)
    print("STAGES BODY:", json.dumps(body3) if isinstance(body3, list) else body3)


if __name__ == "__main__":
    main()
