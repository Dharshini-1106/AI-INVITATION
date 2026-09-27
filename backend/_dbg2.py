import sys, json
sys.stdout.reconfigure(encoding="utf-8")
from app.core.understanding.parser import parse_invitation
from app.core.language import detect_language

annual = """EXCELLENCE 145 OF ACADEMIC AMERICAN THE COLLEGE MADURAI EDUCATION CHARACTER LEADERSHIP SERVICE heritage shaping future since 1881 PURIFICATUS NON CONSUMPTUS Since 1881 DHO Let Knowledge Flourish THE AMERICAN COLLEGE ANNUAL DAY SPORTS 2026 RISE REFLECT REJOICE PLAY EXCEL PERSEVERE An evening celebrate journey People possibilities A Healthier Tomorrow ART FASTER CULTURE CREATIVITY HIGHER BEYOND LIMITS STRONGER TOGETHER Track & Field Events Cultural Performances Awards & Recognitions Team Sports Student Achievements Fun Events for All ANNUAL DAY DATE VENUE 10:00 AM 1:00 PM Friday Tallakulam Madurai 625002 Tamil Nadu India SPORTS DAY 16 October 2026 Tamil Nadu India 2:00 PM - 5:30 PM A Common Purpose Different People STUDENTS TODAY LEADERS TOMORROW DISCIPLINE GROWTH OPPORTUNITY COMMUNITY Trcomnn BMM KUEITEITT mw TTT mML am tr Nual Day 2026 SPORTS"""

res = parse_invitation({}, annual)
print("PEOPLE:", res.get("people"))
print("VENUE:", res.get("venue"))
print("ADDR:", res.get("address"))
print("E0:", res["events"][0].get("time"), res["events"][0].get("end_time"), res["events"][0].get("address") or res.get("address"))
