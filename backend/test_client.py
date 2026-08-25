"""Test the analyze route using FastAPI TestClient to capture the real error."""
import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def make_test_image():
    img = np.full((600, 800, 3), 255, dtype=np.uint8)
    cv2.putText(img, "WEDDING INVITATION", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    cv2.putText(img, "Date: 15/03/2025", (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    cv2.putText(img, "Venue: Green Palace", (150, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def main():
    data = make_test_image()
    resp = client.post(
        "/api/v1/analyze",
        files={"file": ("invitation.png", data, "image/png")},
    )
    print("STATUS:", resp.status_code)
    print("BODY:", resp.text[:2000])


if __name__ == "__main__":
    main()
