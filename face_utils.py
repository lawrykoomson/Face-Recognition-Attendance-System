import face_recognition
import numpy as np
import pickle
import os
import csv
from datetime import datetime

DATA_FILE = "known_faces.pkl"
ATTENDANCE_FILE = "attendance.csv"

# How similar a detected face must be to a known encoding to count as
# a match. Lower = stricter. face_recognition's own default is 0.6.
MATCH_TOLERANCE = 0.5


def _detect_faces(image):
    """
    Find face locations in `image`, trying the fast 'hog' model first
    and falling back to the slower but more accurate 'cnn' model if
    hog finds nothing. This matters in practice: hog can miss faces
    under strong overhead lighting or heavy shadow, which is common
    in webcam photos taken indoors — cnn handles those cases much
    more reliably, at the cost of a few extra seconds per photo.
    """
    locations = face_recognition.face_locations(image, model="hog")
    if not locations:
        locations = face_recognition.face_locations(
            image, model="cnn", number_of_times_to_upsample=1
        )
    return locations


def load_known_faces():
    """Load the stored (name, encoding) pairs. Returns empty lists if
    nothing has been registered yet."""
    if not os.path.exists(DATA_FILE):
        return {"names": [], "encodings": []}

    with open(DATA_FILE, "rb") as f:
        return pickle.load(f)


def save_known_faces(data):
    with open(DATA_FILE, "wb") as f:
        pickle.dump(data, f)


def register_face(image, name):
    """
    Detect a face in `image` (a numpy array, RGB) and store its encoding
    under `name`. Returns (success: bool, message: str).

    A person can be registered multiple times with different photos —
    each photo adds one more encoding, which improves recognition
    accuracy under different lighting/angles.
    """
    face_locations = _detect_faces(image)

    if len(face_locations) == 0:
        return False, "No face detected in this photo. Try a clearer, front-facing photo."

    if len(face_locations) > 1:
        return False, (
            f"Found {len(face_locations)} faces in this photo. "
            "Please use a photo with only one person."
        )

    encoding = face_recognition.face_encodings(image, known_face_locations=face_locations)[0]

    data = load_known_faces()
    data["names"].append(name)
    data["encodings"].append(encoding)
    save_known_faces(data)

    return True, f"✅ Registered {name} successfully."


def registered_names():
    """Unique list of everyone currently registered (a person may have
    multiple encodings from multiple photos)."""
    data = load_known_faces()
    return sorted(set(data["names"]))


def recognize_faces(image):
    """
    Detect every face in `image` and match each against the known
    encodings. Returns a list of dicts:
    [{"name": str, "location": (top, right, bottom, left), "confidence": float}, ...]

    Unrecognized faces are labeled "Unknown" rather than dropped, so
    the app can still show that *someone* was detected, just not
    matched to anyone registered.
    """
    data = load_known_faces()
    known_encodings = data["encodings"]
    known_names = data["names"]

    face_locations = _detect_faces(image)
    face_encodings = face_recognition.face_encodings(image, known_face_locations=face_locations)

    results = []
    for location, encoding in zip(face_locations, face_encodings):
        name = "Unknown"
        confidence = 0.0

        if known_encodings:
            distances = face_recognition.face_distance(known_encodings, encoding)
            best_match_index = int(np.argmin(distances))
            best_distance = distances[best_match_index]

            if best_distance <= MATCH_TOLERANCE:
                name = known_names[best_match_index]
                confidence = round((1 - best_distance) * 100, 1)

        results.append({"name": name, "location": location, "confidence": confidence})

    return results


def mark_attendance(name):
    """
    Log `name` as present right now, unless they've already been
    marked present today (prevents duplicate entries if the same
    photo/session recognizes them more than once).

    Returns True if a new attendance record was added, False if they
    were already marked today.
    """
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    file_exists = os.path.exists(ATTENDANCE_FILE)

    if file_exists:
        with open(ATTENDANCE_FILE, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["Name"] == name and row["Date"] == today:
                    return False  # already marked today

    with open(ATTENDANCE_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Name", "Date", "Time"])
        writer.writerow([name, today, time_str])

    return True


def load_attendance():
    """Return all attendance records as a list of dicts. Empty list if
    nothing's been logged yet."""
    if not os.path.exists(ATTENDANCE_FILE):
        return []

    with open(ATTENDANCE_FILE, "r", newline="") as f:
        return list(csv.DictReader(f))