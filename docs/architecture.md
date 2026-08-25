# System Architecture

## Overview

This document describes the architecture of the **AI-Driven Invitation Understanding
& Adaptive Travel Planning System** — Phase 1: Intelligent Invitation Understanding
and Information Extraction.

The system follows a **client-server** architecture:

- **Frontend (React + Vite)**: mobile-first web UI for uploading/scannning invitations,
  displaying processing progress, and showing structured results.
- **Backend (Python FastAPI)**: REST API that orchestrates the AI pipeline and returns
  structured JSON.

---

## Frontend Architecture

```
src/
├── main.jsx                 # React entry point (BrowserRouter)
├── App.jsx                  # Route definitions
├── config/apiConfig.js      # API base URL + endpoints
├── theme/colors.js          # Global color theme
├── models/InvitationResult.js  # Result model classes
├── services/
│   ├── api.js               # Axios client (health, analyze, stages)
│   └── imageService.js      # File read/preview helpers
├── screens/
│   ├── SplashScreen.jsx     # Splash with auto-navigate
│   ├── HomeScreen.jsx       # Two action cards + backend status
│   ├── GalleryUploadScreen.jsx  # Upload + preview
│   ├── CameraScanScreen.jsx     # Camera capture + preview
│   ├── ProcessingScreen.jsx     # Live progress + API call
│   └── ResultScreen.jsx     # Structured result display
└── components/
    ├── LoadingSpinner.jsx
    ├── ErrorBanner.jsx
    └── ImagePreviewCard.jsx
```

### Data Flow (Frontend)

1. User selects/captures an image → `imageService.prepareFile()` produces a preview.
2. User confirms → navigate to `/processing` with `{ file, previewUrl }`.
3. `ProcessingScreen` calls `analyzeInvitation(file)` (multipart POST).
4. While waiting, it animates through pipeline stages fetched from backend.
5. On success → navigate to `/result` with `{ result, previewUrl }`.
6. `ResultScreen` renders the structured `InvitationResult`.

---

## Backend Architecture

```
backend/app/
├── main.py                  # FastAPI app + CORS + lifespan
├── config.py                # Settings (model toggles, thresholds)
├── schemas/invitation.py    # Pydantic models
├── api/routes/
│   ├── health.py            # GET /api/v1/health
│   └── analyze.py           # POST /api/v1/analyze
├── core/
│   ├── pipeline.py          # Orchestrates all stages
│   ├── quality/brisque.py   # BRISQUE quality scoring
│   ├── enhancement/enhance.py  # CLAHE, deblur, denoise, super-res, perspective
│   ├── detection/layout.py  # DocLayout-YOLO layout analysis
│   ├── ocr/ocreader.py      # PaddleOCR PP-OCRv5
│   ├── language.py          # Language detection
│   ├── postprocess/correction.py  # Sentence-BERT correction
│   └── understanding/parser.py  # LayoutLMv3 + rule-based parser
└── utils/image_utils.py     # Image loading/orientation helpers
```

### AI Pipeline (backend/core/pipeline.py)

```
Image bytes
   │
   ▼
1. BRISQUE Quality Analysis
   ├─ blur / noise / brightness / rotation / resolution
   ▼
2. Enhancement (if needed)
   ├─ CLAHE (contrast)
   ├─ Denoise (NLM)
   ├─ Deblur (unsharp)
   ├─ Perspective Correction
   └─ Super-Resolution (FSRCNN / bicubic)
   ▼
3. Layout Analysis (DocLayout-YOLO / rule-based)
   ▼
4. Multilingual OCR (PaddleOCR PP-OCRv5)
   ▼
5. Language Detection (script ranges + OCR hints)
   ▼
6. OCR Correction (Sentence-BERT / dictionary)
   ▼
7. Understanding & Structuring (LayoutLMv3 + rule-based parser)
   ▼
Structured JSON result
```

### Graceful Fallback Design

Each heavy AI module (DocLayout-YOLO, PaddleOCR, Sentence-BERT, LayoutLMv3) is
wrapped in a try/except that falls back to a lightweight rule-based implementation
when the model/weights are unavailable. This ensures the system runs out-of-the-box
while remaining production-ready when models are installed.

---

## Data Model

### InvitationResult (Pydantic / JSON)

| Field | Type | Description |
|-------|------|-------------|
| event_name | string | Primary event name |
| event_type | string | Wedding / Reception / Birthday / etc. |
| bride_name | string | Bride name |
| groom_name | string | Groom name |
| date | string | Event date |
| time | string | Event time |
| venue | string | Venue name |
| address | string | Venue address |
| contact_number | string | Contact phone |
| language | string | Detected language |
| confidence_score | float | 0..1 overall confidence |
| number_of_events | int | Event count |
| events | Event[] | Multiple events (if any) |
| quality | ImageQuality | Quality analysis details |
| raw_text | string | Raw OCR text |
| processing_notes | string[] | Pipeline notes |

---

## Modularity for Future Phases

The pipeline is split into isolated, replaceable modules. Future phases can add:

- **Google Calendar Integration** — new route/service that consumes `InvitationResult.date/time`
- **Google Maps** — new route/service that consumes `InvitationResult.venue/address`
- **Venue Validation** — new post-processing step
- **Real-time Traffic Analysis** — new service
- **Adaptive Leave-Time Notifications** — new service

All without modifying the existing invitation understanding module (`core/pipeline.py`
and its sub-modules), because each stage exposes a clean interface and returns
structured data.

---

## Security & Performance

- **CORS**: restricted to `localhost:5173` / `localhost:3000`
- **Upload limits**: max 20 MB per file
- **File type validation**: JPG/PNG/BMP/WEBP/TIFF/PDF
- **Async requests**: FastAPI async endpoints; heavy models loaded lazily
- **Timeout**: 180s client-side timeout for the AI pipeline
