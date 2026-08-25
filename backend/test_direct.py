"""Run the pipeline directly to surface any errors in fallback mode."""
import cv2
import numpy as np
from app.core.pipeline import run_pipeline


def make_test_image():
    img = np.full((600, 800, 3), 255, dtype=np.uint8)
    cv2.putText(img, "WEDDING INVITATION", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    cv2.putText(img, "Mr. Arjun & Ms. Priya", (120, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(img, "Date: 15/03/2025", (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    cv2.putText(img, "Time: 6:30 PM", (150, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    cv2.putText(img, "Venue: Green Palace Convention Hall", (150, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "Contact: 9876543210", (150, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    return img


def main():
    img = make_test_image()
    ok, buf = cv2.imencode(".png", img)
    data = buf.tobytes()
    try:
        result = run_pipeline(data, "test.png")
        print("PIPELINE OK")
        print("language:", result["language"])
        print("date:", result["date"])
        print("time:", result["time"])
        print("num_events:", result["number_of_events"])
        print("notes:", result["processing_notes"])
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
