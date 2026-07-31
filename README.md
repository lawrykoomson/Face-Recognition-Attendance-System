# 🎓 AI Face Recognition Attendance System

An interactive, application-based attendance system that recognizes registered
students from a photo and logs their attendance automatically — no manual
sign-in sheets, no verbal roll call.

This is a from-scratch rebuild of an earlier group project (originally a
notebook/script-based implementation), reimplemented as a full interactive
application with proper error handling, a two-tier detection strategy for
real-world lighting conditions, and an exportable attendance log.

## Features

- **Register students** by webcam or photo upload — supports multiple
  reference photos per person for improved recognition accuracy
- **Take attendance** from a live webcam capture or an uploaded photo, with
  recognized faces drawn back onto the image (green box = matched, red box =
  unrecognized)
- **Two-tier face detection** — tries a fast detector first, and
  automatically falls back to a slower, more accurate model if the fast one
  finds nothing. This was added after real testing showed the fast model
  missing faces under strong indoor overhead lighting.
- **Duplicate-safe attendance logging** — a student marked present once
  today won't be logged twice, even if recognition is re-run on a retake
- **Attendance log** with date filtering and CSV export
- **Clear error handling** for common failure cases: no face detected,
  multiple faces in a registration photo, unrecognized faces

## How It Works

```
📸 Register: Photo → face detection → face encoding → stored
        ↓
✅ Take Attendance: Photo → face detection → compare against all
   stored encodings → best match → logged with timestamp
   (skipped if already marked present today)
        ↓
📋 Attendance Log: view, filter by date, export as CSV
```

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Face detection & recognition | face_recognition (built on dlib's ResNet-based face embedding model) |
| Image handling | OpenCV, Pillow |
| Data storage | pickle (face encodings), CSV (attendance records) |
| Language | Python 3 |

## Project Structure

```
attendance-system/
├── app.py            # Streamlit UI — registration, attendance, log views
├── face_utils.py       # Core face detection, encoding, matching, and logging logic
└── requirements.txt      # Python dependencies
```

## Running Locally

`face_recognition` depends on `dlib`, which requires C++ build tools to
compile on Windows — this is the most common setup obstacle. If you hit
install errors, ensure CMake and Visual Studio Build Tools are installed
before running `pip install face_recognition`.

```bash
git clone https://github.com/lawrykoomson/Face-Recognition-Attendance-System.git
cd Face-Recognition-Attendance-System
pip install -r requirements.txt
streamlit run app.py
```

## Design Notes

- **Detection strategy**: face detection is the step most sensitive to
  real-world conditions. The app tries dlib's HOG detector first (fast,
  well under a second per photo) and automatically retries with the CNN
  detector (slower, several seconds, but substantially more robust to
  difficult lighting) if HOG finds nothing. This wasn't a speculative
  design choice — it was added after HOG failed to detect a clearly visible
  face under strong overhead lighting during testing.
- **Match tolerance**: set to 0.5 (stricter than the library's default of
  0.6) to reduce false-positive matches, at some cost to recognizing harder
  cases. This is a tunable trade-off between false accepts and false
  rejects.
- **Multiple encodings per person**: registering the same person with
  several photos (different lighting/angles) measurably improves match
  confidence, since matching compares against every stored encoding and
  takes the closest one.
- **Storage**: encodings and attendance records are stored in flat files
  (pickle + CSV) rather than a database, appropriate for a classroom-scale
  deployment; a larger rollout would warrant a proper database for
  concurrent access.

## Author

**Lawrence Koomson**
Data Engineering & Cloud Computing student, University of Cape Coast
[LinkedIn](https://www.linkedin.com/in/lawrence-koomson-689774266/) · [GitHub](https://github.com/lawrykoomson)