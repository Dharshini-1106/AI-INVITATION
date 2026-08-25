# API Documentation

Base URL: `http://localhost:8000`

Interactive docs (Swagger UI): `http://localhost:8000/docs`

---

## GET /api/v1/health

Health check endpoint.

### Response

```json
{
  "status": "ok",
  "app_name": "Invitation Understanding API",
  "version": "1.0.0"
}
```

---

## GET /api/v1/pipeline/stages

Returns the ordered list of pipeline stage names (used by the frontend progress UI).

### Response

```json
[
  "Analyzing image quality (BRISQUE)",
  "Enhancing image (CLAHE/Deblur/Denoise/Super-Resolution)",
  "Detecting invitation layout",
  "Performing multilingual OCR (PP-OCRv5)",
  "Correcting OCR mistakes (Sentence-BERT)",
  "Understanding invitation context (LayoutLMv3)",
  "Structuring extracted information"
]
```

---

## POST /api/v1/analyze

Uploads an invitation image (or PDF) and returns the structured extraction result.

### Request

- **Content-Type**: `multipart/form-data`
- **Field**: `file` — the image file
- **Allowed types**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tiff`, `.pdf`
- **Max size**: 20 MB

### Example (curl)

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -F "file=@invitation.jpg"
```

### Response

```json
{
  "event_name": "Wedding Invitation",
  "event_type": "Wedding",
  "bride_name": "Priya",
  "groom_name": "Arun",
  "date": "12/12/2025",
  "time": "10:30 AM",
  "venue": "Sundaram Mahal",
  "address": "123 Anna Nagar, Chennai",
  "contact_number": "+91 98765 43210",
  "language": "Tamil",
  "confidence_score": 0.87,
  "number_of_events": 1,
  "events": [
    {
      "event_name": "Wedding Invitation",
      "event_type": "Wedding",
      "bride_name": "Priya",
      "groom_name": "Vijay",
      "date": "12/12/2025",
      "time": "10:30 AM",
      "venue": "Sundaram Mahal",
      "address": "123 Anna Nagar, Chennai",
      "contact_number": "+91 98765 43210",
      "confidence": 0.87
    }
  ],
  "quality": {
    "score": 82.0,
    "is_blurred": false,
    "is_noisy": false,
    "is_dark": false,
    "is_bright": false,
    "is_rotated": false,
    "low_resolution": false,
    "needs_enhancement": false,
    "applied_enhancements": []
  },
  "raw_text": "...",
  "processing_notes": [
    "Detected language: Tamil",
    "Total processing time: 3.45s"
  ]
}
```

### Error Responses

| Status | Description |
|--------|-------------|
| 400 | Unsupported file type / empty file |
| 413 | File too large (> 20 MB) |
| 500 | Pipeline/analysis failure |

---

## Frontend Integration

The React frontend calls these endpoints via `src/services/api.js`:

```js
import { analyzeInvitation, getPipelineStages, checkHealth } from './services/api';

// Health check
const health = await checkHealth();

// Get pipeline stages
const stages = await getPipelineStages();

// Analyze a file
const result = await analyzeInvitation(file);
