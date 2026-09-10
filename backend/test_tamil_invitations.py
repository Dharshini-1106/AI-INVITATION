"""Test extraction against the 6 provided Tamil invitations.

Run: python test_tamil_invitations.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.understanding.parser import parse_invitation  # noqa: E402


def check(label, raw, expected, fields=("event_type", "bride_name", "groom_name",
                                        "date", "time", "venue", "address",
                                        "contact_number")):
    r = parse_invitation({}, raw)
    fails = []
    for f in fields:
        got = r.get(f, "")
        exp = expected.get(f, "")
        if got != exp:
            fails.append(f"{f}={got!r} (exp {exp!r})")
    status = "OK" if not fails else "FAIL"
    print(f"[{status}] {label}")
    for fl in fails:
        print(f"        {fl}")
    if status == "FAIL":
        print(f"    full result: { {k: r.get(k, '') for k in fields} }")
    return status == "OK"


def main():
    tests = [
        # Invitation 1: Tamil Hindu wedding - Rajesh & Kamala Iyer's son
        ("tamil_hindu_wedding_rajesh",
         ("ராஜேஷ்\n"
          "எஸ்/ ஓ இருமதி. கம்லா ஐயர்\n"
          "புதன்\n"
          "பிரியங்கா\n"
          "டி/ ஓ இருமதி. வித்யா நாயக்\n"
          "இரு. மோகன் நாயக், கர்நாடகா\n"
          "திருமண அழைப்பிதழ்\n"
          "மண்ணோடு விதை சேர்ந்து\n"
          "மணத்தோடு மலர் பூத்தது போல்\n"
          "மனதோடு மனம் சேர்ந்து\n"
          "மணம் வீசும் இம்மங்கல திருமணநாளில்\n"
          "மகத்தான வாழ்வில் மாலையிட்டு\n"
          "மனதை பரிமாறும் வேளையில்\n"
          "அன்பினால் உரமிட்டு\n"
          "இவ்விதைக்கு வளம் சேர்க்க\n"
          "உங்கள் அனைவரையும் எங்கள் திருமண நன்னாளில் கலந்து\n"
          "கொள்ள அன்புடன் அழைக்கிறோம்...\n"
          "செப்டம்பர் 17, 2025\n"
          "செவ்வாய்கிழமை, காலை 10 மணிக்கு\n"
          "இடம்\n"
          "தாமரை நீதிமன்ற திருமண மண்டபம்,\n"
          "சாஸ்திரி நகர, லக்ஷ்மி பூங்கா, கர்நாடகா\n"
          "அனுப்புநரின் பெயர்\n"
          "திரு விக்ரம் ஐயர் மற்றும் குடும்பம்\n"
          "9910123456 | 1234567891"),
         {"event_type": "Wedding", "bride_name": "பிரியங்கா", "groom_name": "ராஜேஷ்",
          "date": "September 17, 2025", "time": "10:00 AM",
          "venue": "தாமரை நீதிமன்ற திருமண மண்டபம்",
          "address": "சாஸ்திரி நகர, லக்ஷ்மி பூங்கா, கர்நாடகா",
          "contact_number": "9910123456"}),

        # Invitation 2: Tamil Nikah - A. Muhammudu Abitha & S. Muhammad Mushtak
        ("tamil_nikah",
         ("கஸ்லாதலிர் ரஹ்மானிர் ரஹிம்\n"
          "திருமண (நickersாஹ்) அழைப்பிதழ்\n"
          "அன்புடையீர், அஸ்ஸலாமு அலைக்கும் (வரஹ்)\n"
          "எல்லாம் வல்ல அல்லாஹ்வின் பேரருளாலும், நபிகள் நாயகம் (ஸல்) அவர்களின்\n"
          "துஆ பரக்கத்தாலும், நிகழும் ஹிஜ்ரி 1448ம் ஆண்டு, ரபியுல் அவ்வல் பிறை 09, 23-08-2026\n"
          "ஞாயிற்றுக்கிழமை காலை 11.30 மணிக்கு\n"
          "காயாமொழி (மர்ஹும்) அல்ஹாஜ் S.S.முகைதீன் தம்பி - ஹாஜிமா ஆபிதா\n"
          "குறும்பூர் (மர்ஹும்) A.அப்துல் அலி - ஹாஜிமா. சகு பாத்திமா\n"
          "ஆகியோரின் பேத்தியும், எங்களின் புதல்வி\n"
          "A. முஹம்மது ஆபிதா M.A.,\n"
          "மணமகளுக்கும்\n"
          "காயாமொழி (மர்ஹும்) ஹாஜி S.அப்துல் காதர் - ஹாஜிமா கம்சா பீவி\n"
          "உடன்குடி, புதுமனை அல்ஹாஜ் மஹ்மூது காசிம் - ஜனாபா ரஹ்மத் நிஸா\n"
          "ஆகியோரின் பேரனும்\n"
          "ஜனாப் A.சலாகுதின் - ஜனாபா S.நபிஸா முஹானிரா\n"
          "ஆகியோரின் புதல்வன்\n"
          "S. முஹம்மது முஷ்தாக் B.E.,\n"
          "மணமகனுக்கும்\n"
          "திருமணம் (நickersாஹ்) செய்ய பெரியோர்களால் நிச்சயித்தவண்ணம், இன்ஷா அல்லாஹ்,\n"
          "காயங்குறிச்சி அரஸ் திருமண மஹாலில் நடைபெறும் நickersாஹ்விற்கும் அதனை\n"
          "தொடர்ந்து நடைபெறும் வலிமா விருந்திலும் கலந்து கொண்டு சிறப்பிக்குமாறு அன்புடன்\n"
          "அழைக்கின்றோம்.\n"
          "தங்கள் நல்வரவை இனிதே விரும்பும்\n"
          "ஆயிஷா ஆபிதா\n"
          "தங்கள் அன்புடன்\n"
          "ஜனாப் M.T.முஹம்மது அன்வர் ஹூசைன் B.A., B.L.,\n"
          "ஜனாபா A.சித்தி ஹாஜர்\n"
          "8/9A, பள்ளிவாசல் தெரு, காயாமொழி\n"
          "போன் : 8884950410, 7200853041"),
         {"event_type": "Nikah", "bride_name": "A. முஹம்மது ஆபிதா", "groom_name": "S. முஹம்மது முஷ்தாக்",
          "date": "August 23, 2026", "time": "11:30 AM",
          "venue": "காயங்குறிச்சி அரஸ் திருமண மஹாலில்",
          "address": "காயாமொழி",
          "contact_number": "8884950410"}),

        # Invitation 3: Tamil Hindu wedding - UK venue
        ("tamil_hindu_wedding_uk",
         ("சிவயம்\n"
          "திருமண அழைப்பிதழ்\n"
          "அன்புடையீர்,\n"
          "நிகழும் மங்களகரமான சோபகிருது வருடம் கார்த்திகை மாதம்\n"
          "3-ம் நாள் (19-11-2023) ஞாயிற்றுக்கிழமை சப்தமிதியும்,\n"
          "திருவோண நட்சத்திரமும் அமிர்த யோகமும் கூடிய பகல் 10-15\n"
          "முதல் 11-45 மணிவரையுள்ள மகாலக்ஷ்மியின் சுப முகூர்த்த\n"
          "வேளையில்\n"
          "எமது புத்திரன்\n"
          "திருநிறைச்செல்வன்\n"
          "விஷ்ணு\n"
          "அவர்களுக்கும்\n"
          "எமது புத்திரி\n"
          "திருநிறைச்செல்வி\n"
          "அஞ்சலி\n"
          "அவர்களுக்கும்\n"
          "இறைவன் திருவருள் துணைகொண்டு திருமாங்கல்யதாரணம்\n"
          "செய்து வைக்க பெரியோர்கள் நிச்சயித்திருப்பதால் அத்தருணம்\n"
          "தாங்கள் தங்கள் குடும்ப சமேதராய் வருகை தந்து\n"
          "மணமக்களை ஆசீர்வதித்து தொடர்ந்து நடைபெறும்\n"
          "விருந்துபசாரத்திலும் கலந்து சிறப்பிக்குமாறு அன்புடன்\n"
          "அழைக்கின்றோம்\n"
          "திருமணம் நடைபெறும் இடம்\n"
          "The Manor House, West St, Chippenham, SN14 7HX\n"
          "இந்நாள் தங்கள் நல்வரவை நாடும்\n"
          "திரு. திருமதி. பிரபாகரன்\n"
          "65 Manor Road\n"
          "Islington\n"
          "London\n"
          "N51 9VK\n"
          "திரு. திருமதி. விஸ்வநாதன்\n"
          "23 Grange Road\n"
          "Ealing\n"
          "London\n"
          "W54 2BY\n"
          "(No boxed gifts please)"),
         {"event_type": "Wedding", "bride_name": "அஞ்சலி", "groom_name": "விஷ்ணு",
          "date": "November 19, 2023", "time": "10:15 AM",
          "venue": "The Manor House, West St, Chippenham, SN14 7HX",
          "address": "Chippenham, SN14 7HX",
          "contact_number": ""}),

        # Invitation 4: Tamil birthday invitation
        ("tamil_birthday",
         ("ஆகாஷின்\n"
          "முதல் பிறந்தநாள் விழா\n"
          "அழைப்பிதழ்\n"
          "அக்டோபர் 12, 2025 செவ்வாய்கிழமை\n"
          "பிறந்தநாள் பார்ட்டி\n"
          "கொண்டாட்டத்திற்காக ஷர்மா\n"
          "குடும்பத்தினர் உங்கள்\n"
          "ஆசீர்வாதங்களைத் தேடுகிறார்கள்\n"
          "உங்கள் இருப்பே மிகவும்\n"
          "விலையுயர்ந்த பரிசு.\n"
          "இடம்\n"
          "லேக் வே எஸ்டேட், சஹேலி மார்க்,\n"
          "உதய்பூர்\n"
          "நிமந்தரக்\n"
          "ராகேஷ் சர்மா மற்றும் குடும்பம்\n"
          "91919-91919 - 81818-18181"),
         {"event_type": "Birthday", "bride_name": "", "groom_name": "",
          "date": "October 12, 2025", "time": "",
          "venue": "லேக் வே எஸ்டேட், சஹேலி மார்க்",
          "address": "உதய்பூர்",
          "contact_number": "91919-91919"}),

        # Invitation 5: Tamil baby shower (வளைக்காப்பு)
        ("tamil_baby_shower",
         ("வளைக்காப்பு விழா அழைப்பிதழ்\n"
          "திரு. சிவக்குமார் - திருமதி லலிதா\n"
          "அவர்களின் மருமகள்\n"
          "அன்புள்ள குடும்பத்தினரும் நண்பர்களும்,\n"
          "எங்கள் இனிய வளைக்காப்பு விழாவிற்கு\n"
          "அன்புடன் அழைக்கிறோம்.\n"
          "திருமதி கண்மணி\n"
          "நாள்: 23-11-2025. ஞாயிற்றுக்கிழமை\n"
          "நேரம்: காலை 10:30 மணி\n"
          "இடம்: - ஸ்ரீ பல்லவி ஹால்,\n"
          "662/585, திருவொற்றியூர் ஹை ரோடு,\n"
          "சென்னை."),
         {"event_type": "Baby Shower", "bride_name": "", "groom_name": "",
          "date": "November 23, 2025", "time": "10:30 AM",
          "venue": "ஸ்ரீ பல்லவி ஹால்",
          "address": "662/585, திருவொற்றியூர் ஹை ரோடு, சென்னை",
          "contact_number": ""}),

        # Invitation 6: Bilingual Tamil-English wedding - V. Ashwini & C. Murugan
        ("tamil_english_wedding_bilingual",
         ("|| Sri Venkateshaperumal Thunai ||\n"
          "Smt. V. Parvathi & Sri N. Vasu\n"
          "திருமண அழைப்பிதழ்\n"
          "Smt. Shamala & Late Sri Chandran\n"
          "Aanuppu Village, Gudiyatham Taluk, Vellore Dist.\n"
          "Solicit your gracious presence with family and friends\n"
          "on the auspicious occasion of the marriage of\n"
          "Selvi: V. Ashwini\n"
          "(D/o. Smt. V. Parvathi & Sri N. Vasu)\n"
          "Selvan: C. Murugan\n"
          "(S/o. Smt. Shamala & Late Sri Chandran)\n"
          "Sri Venkateshwara Electricals, Bangalore\n"
          "on Sunday, 14th February 2016 at\n"
          "Siddalingeshwara Kalyana Mantapa\n"
          "Kurubarahalli, Kaveri Nagar, Shivan Temple, J.C. Nagar,\n"
          "Bangalore - 560 086.\n"
          "Reception:\n"
          "13-02-2016, Saturday\n"
          "7:00 pm onwards\n"
          "Lagnam:\n"
          "Kumbha\n"
          "14-02-2016, Sunday\n"
          "6:00 am to 7:30 am\n"
          "Muhurtham:\n"
          "With Best Compliments From: Relatives & Friends"),
         {"event_type": "Wedding", "bride_name": "V. Ashwini", "groom_name": "C. Murugan",
          "date": "February 14, 2016", "time": "6:00 AM",
          "venue": "Siddalingeshwara Kalyana Mantapa",
          "address": "Kurubarahalli, Kaveri Nagar, Shivan Temple, J.C. Nagar, Bangalore - 560 086",
          "contact_number": ""}),
    ]

    passed = 0
    for label, raw, exp in tests:
        if check(label, raw, exp):
            passed += 1
    print(f"\n{passed}/{len(tests)} Tamil invitation tests passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
