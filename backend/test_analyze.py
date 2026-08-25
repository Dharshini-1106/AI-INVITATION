"""End-to-end test of the /analyze endpoint with a synthetic invitation image."""
import json
import io
import urllib.request
import uuid

import cv2
import numpy as np


def make_test_image():
    """Create a synthetic invitation image with text-like regions."""
    img = np.full((600, 800, 3), 255, dtype=np.uint8)
    # Title band
    cv2.putText(img, "WEDDING INVITATION", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    cv2.putText(img, "Mr. Arjun & Ms. Priya", (120, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(img, "request the honour of your presence", (140, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 60), 1)
    cv2.putText(img, "Date: 15/03/2025", (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    cv2.putText(img, "Time: 6:30 PM", (150, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    cv2.putText(img, "Venue: Green Palace Convention Hall", (150, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "Address: Anna Nagar, Chennai", (150, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "Contact: 9876543210", (150, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    return img


def main():
    img = make_test_image()
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        print("FAILED to encode test image")
        return
    data = buf.tobytes()

    boundary = uuid.uuid4().hex
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="invitation.png"\r\n'
        f"Content-Type: image/png\r\n\r\n"
    ).encode("utf-8") + data + f"\r\n--{boundary}--\r\n".encode("utf-8")

    req = urllib.request.Request(
        "http://127.0.0.1:8000/api/v1/analyze",
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            print("ANALYZE STATUS:", resp.status)
            print("LANGUAGE:", result.get("language"))
            print("EVENT NAME:", result.get("event_name"))
            print("BRIDE:", result.get("bride_name"))
            print("GROOM:", result.get("groom_name"))
            print("DATE:", result.get("date"))
            print("TIME:", result.get("time"))
            print("VENUE:", result.get("venue"))
            print("ADDRESS:", result.get("address"))
            print("CONTACT:", result.get("contact_number"))
            print("CONFIDENCE:", result.get("confidence_score"))
            print("NUM EVENTS:", result.get("number_of_events"))
            print("QUALITY:", json.dumps(result.get("quality")))
            print("NOTES:", result.get("processing_notes"))
    except Exception as exc:  # noqa: BLE001
        print("ANALYZE FAILED:", repr(exc))


if __name__ == "__main__":
    main()
