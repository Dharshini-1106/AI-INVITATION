# AI-Driven Invitation Understanding & Adaptive Travel Planning System

## Phase 1 — Intelligent Invitation Understanding and Information Extraction

A mobile-first web application that lets users upload or scan invitation cards and
AI extracts all important event information accurately — across multilingual,
decorative, and noisy invitations.

> **Note:** This phase focuses **only** on invitation understanding and information
> extraction. Calendar creation, reminders, Google Maps, travel planning, and
> notifications are **not** included — the architecture is modular so future phases
> can add them without changing the existing invitation understanding module.

---

## ✨ Features

- **6 screens**: Splash, Home, Upload from Gallery, Scan with Camera, Processing, Result
- **Multilingual OCR**: English, Tamil, Hindi, Malayalam, Telugu, and mixed-language
- **Decorative & cursive font support**: wedding fonts, curved text, colored text
- **Robust image handling**: blurred, low-res, noisy, rotated, shadow, dark, bright, skewed
- **Automatic image enhancement**: CLAHE, perspective correction, super-resolution, denoising, deblurring
- **Structured output**: event name, bride/groom, date, time, venue, address, contact, language, confidence
- **Multi-event detection**: identifies multiple events on a single invitation
- **Live processing progress** with stage-by-stage updates
- **Live preview** before processing
- **Graceful degradation**: heavy AI models auto-skip to rule-based fallbacks if weights aren't downloaded

---

## 🏗️ Architecture

```
┌─────────────────────────┐        ┌──────────────────────────────┐
│       React Frontend     │  HTTP  │       FastAPI Backend         │
│  (Vite + React Router)   │ ─────► │  /api/v1/analyze             │
│  Upload / Camera / Screens│        │                              │
└─────────────────────────┘        └──────────────┬───────────────┘
                                                  │
                                    ┌─────────────▼──────────────┐
                                    │       AI Pipeline          │
                                    │  1. BRISQUE quality       │
                                    │  2. Enhancement (CLAHE...) │
                                    │  3. DocLayout-YOLO layout  │
                                    │  4. PaddleOCR PP-OCRv5     │
                                    │  5. Language detection     │
                                    │  6. Sentence-BERT correct  │
                                    │  7. LayoutLMv3 + parser    │
                                    └────────────────────────────┘
```

### Tech Stack
| Layer | Technology |
|-------|-----------|
| Frontend | React 18, Vite, React Router, Axios |
| Backend | Python FastAPI, Uvicorn |
| AI Models | BRISQUE, CLAHE, DocLayout-YOLO, PaddleOCR (PP-OCRv5), Sentence-BERT, LayoutLMv3 |

---

## 📁 Project Structure

```
d:/MINI/
├── README.md
├── docs/
│   ├── architecture.md
│   └── api.md
├── backend/
│   ├── requirements.txt
│   ├── run.py
│   └── app/
│       ├── main.py
│       ├── config.py
│       ├── schemas/
│       ├── api/routes/
│       ├── core/
│       │   ├── pipeline.py
│       │   ├── quality/brisque.py
│       │   ├── enhancement/enhance.py
│       │   ├── detection/layout.py
│       │   ├── ocr/ocreader.py
│       │   ├── language.py
│       │   ├── postprocess/correction.py
│       │   └── understanding/parser.py
│       └── utils/
└── frontend/
    ├── package.json
    ├── vite.config.js
    └── src/
        ├── main.jsx
        ├── App.jsx
        ├── config/
        ├── theme/
        ├── models/
        ├── services/
        ├── screens/
        └── components/
```

---

## 🚀 Getting Started

### 1. Backend Setup

```bash
cd backend
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
python run.py
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

> **Heavy model note:** The first OCR/ML model load will download weights. If you
> want a lightweight run without models, the app automatically falls back to
> rule-based heuristics.

### 2. Frontend (Web) Setup

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

### 3. Mobile App (Expo) Setup

The mobile app is an **Expo (React Native)** project that runs the same screens as
the web frontend. It supports **web** and **native (Android)** — no Android Studio
required.

```bash
cd mobile
npm install

# Run in the browser (quick testing)
npm run web

# Run on a phone via LAN (Expo Go or dev build)
npm start
```

**To build an installable APK without Android Studio:**
Expo builds the APK on the cloud via **EAS Build** (no local Android Studio needed):

```bash
cd mobile
npm install -g eas-cli
eas login
eas build -p android --profile preview
```

Download the generated `.apk` from the EAS dashboard and install it on your
Android phone.

> **Backend connection (auto-discovery):** The mobile app now **automatically finds
> the backend**. `mobile/src/services/api.js` probes candidate base URLs and uses the
> first one whose `/health` returns `ok`. Candidates tried:
>   1. `EXPO_PUBLIC_API_URL` env override (highest priority).
>   2. Your PC's **LAN IP**, auto-derived from Expo's dev-server `hostUri` — this is
>      how a real phone running via Expo Go / dev build reaches your PC's backend.
>   3. `http://localhost:8000/api/v1` (web browser on PC, or Android emulator with
>      `adb reverse tcp:8000 tcp:8000`).
>
> **Requirements for a real phone to reach the backend:**
>   - The backend must run on `0.0.0.0` (run.py already does).
>   - The phone and PC must be on the **same WiFi** --OR-- use USB port-forwarding:
>     `adb reverse tcp:8000 tcp:8000`.
>   - Windows firewall must allow inbound TCP on port 8000.
>
> To force a specific address (e.g. a fixed LAN IP), set:
>   `EXPO_PUBLIC_API_URL=http://192.168.1.5:8000/api/v1`
>
> On the Home screen, the app shows the resolved API address, so you can verify the
> phone is actually reaching the backend.

### Troubleshooting "Analysis failed" / "Server error: Network Error"
This error appears when the phone cannot reach the FastAPI backend. To resolve:
1. Start the backend: `cd backend && venv\Scripts\activate && python run.py`
   (or on macOS/Linux: `source venv/bin/activate && python run.py`).
2. Confirm it's healthy: `http://<PC_LAN_IP>:8000/api/v1/health` returns `{"status":"ok",...}`.
3. Make sure the phone is on the **same WiFi** as the PC, or run
   `adb reverse tcp:8000 tcp:8000` for a USB connection.
4. Allow port 8000 through the Windows firewall.
5. Restart the Expo app so it re-runs backend auto-discovery. The Home screen will
   show the backend status and the API address it connected to.

---

## 📡 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/health` | Health check |
| GET | `/api/v1/pipeline/stages` | Pipeline stage names (for progress UI) |
| POST | `/api/v1/analyze` | Upload & analyze an invitation (multipart `file`) |

See [docs/api.md](docs/api.md) for full details.

---

## 🔮 Future Phases (Planned, Not Implemented)

The modular pipeline allows future phases to add:
- Google Calendar Integration
- Google Maps
- Venue Validation
- Real-time Traffic Analysis
- Adaptive Leave-Time Notifications

without modifying the existing invitation understanding module.
