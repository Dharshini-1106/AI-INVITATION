import os, sys, json, requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app.config import settings


def route(origin_addr, dest_addr):
    payload = {
        "origin": {"address": origin_addr},
        "destination": {"address": dest_addr},
        "travelMode": "DRIVE",
        "routingPreference": "TRAFFIC_AWARE",
    }
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": settings.google_maps_api_key,
        "X-Goog-FieldMask": "routes.duration,routes.distanceMeters",
    }
    r = requests.post(
        "https://routes.googleapis.com/directions/v2:computeRoutes",
        json=payload, headers=headers, timeout=30,
    )
    data = r.json()
    for route in data.get("routes", []):
        print("  %r: %sm / %s" % (dest_addr, route["distanceMeters"], route["duration"]))


print("Test 5 - with India:")
route("Thiruchendur, Tamil Nadu",
      "Sivasami Maaligai Marriage Hall, Alangulam Road, Mukkudal, Tirunelveli - 627 758, Tamil Nadu, India")
print("Test 6 - venue + locality only (no street):")
route("Thiruchendur, Tamil Nadu",
      "Sivasami Maaligai, Mukkudal, Tamil Nadu")
print("Test 7 - just Mukkudal:")
route("Thiruchendur, Tamil Nadu",
      "Mukkudal, Tirunelveli, Tamil Nadu")
print("Test 8 - Alangulam Road Mukkudal:")
route("Thiruchendur, Tamil Nadu",
      "Alangulam Road, Mukkudal, Tirunelveli, Tamil Nadu")
print("Test 9 - Sivasami Maaligai + correct pin:")
route("Thiruchendur, Tamil Nadu",
      "Sivasami Maaligai, Mukkudal, Tirunelveli - 627601, Tamil Nadu")
print("Test 10 - venue + full address with correct pin:")
route("Thiruchendur, Tamil Nadu",
      "Sivasami Maaligai, Alangulam Road, Mukkudal, Tirunelveli - 627601, Tamil Nadu")