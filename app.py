
import os
import csv
import re
import base64
import threading
from io import BytesIO
from datetime import datetime
from functools import wraps

import cv2
import numpy as np
import face_recognition
from PIL import Image

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    jsonify,
    session,
    flash,
)


 
# FLASK CONFIGURATION
 

app = Flask(__name__)

# IMPORTANT:
# Change this in production using an environment variable.
app.secret_key = os.getenv(
    "FLASK_SECRET_KEY",
    "development-secret-change-this"
)

# For local testing only.
# In production, set ADMIN_PASSWORD as an environment variable.
ADMIN_PASSWORD = os.getenv(
    "ADMIN_PASSWORD",
    "password123"
)


 
# STORAGE
 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
ATTENDANCE_FILE = os.path.join(BASE_DIR, "attendance.csv")

os.makedirs(DATASET_DIR, exist_ok=True)


# Create attendance file if it doesn't exist
if not os.path.exists(ATTENDANCE_FILE):
    with open(
        ATTENDANCE_FILE,
        mode="w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Name",
            "Date",
            "Time"
        ])


# Used to prevent two requests from writing to CSV simultaneously
attendance_lock = threading.Lock()


 
# FACE DATABASE
 

known_face_encodings = []
known_face_names = []


def load_known_faces():
    """
    Load all student images from dataset/
    and generate face encodings.
    """

    encodings = []
    names = []

    if not os.path.exists(DATASET_DIR):
        return encodings, names

    for filename in os.listdir(DATASET_DIR):

        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            continue

        path = os.path.join(DATASET_DIR, filename)

        try:

            image = face_recognition.load_image_file(path)

            face_encodings = face_recognition.face_encodings(
                image
            )

            # We only accept images containing at least one face
            if len(face_encodings) == 0:
                print(
                    f"[WARNING] No face found in {filename}"
                )
                continue

            # First face is used
            encoding = face_encodings[0]

            name = os.path.splitext(filename)[0]
            name = name.replace("_", " ")

            encodings.append(encoding)
            names.append(name)

            print(f"[FACE LOADED] {name}")

        except Exception as e:

            print(
                f"[ERROR] Could not load {filename}: {e}"
            )

    print(
        f"[DATABASE] Loaded {len(names)} student face(s)"
    )

    return encodings, names


def reload_face_database():
    """
    Reload face database without restarting Flask.
    """

    global known_face_encodings
    global known_face_names

    (
        known_face_encodings,
        known_face_names
    ) = load_known_faces()


# Initial loading
reload_face_database()


 
# ADMIN AUTHENTICATION
 

def admin_required(function):
    """
    Protect admin-only pages.
    """

    @wraps(function)
    def decorated_function(*args, **kwargs):

        if not session.get("admin_authenticated"):
            return redirect(
                url_for("admin_login")
            )

        return function(*args, **kwargs)

    return decorated_function


 
# ATTENDANCE FUNCTIONS
 

def get_attendance_records():
    """
    Read all attendance records from CSV.
    """

    records = []

    if not os.path.exists(ATTENDANCE_FILE):
        return records

    with open(
        ATTENDANCE_FILE,
        mode="r",
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            if not row:
                continue

            records.append({
                "name": row.get("Name", ""),
                "date": row.get("Date", ""),
                "time": row.get("Time", "")
            })

    return records


def mark_attendance(name):
    """
    Mark attendance once per student per day.

    Returns:
        True  -> newly marked
        False -> already marked today
    """

    now = datetime.now()

    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    with attendance_lock:

        records = get_attendance_records()

        for record in records:

            if (
                record["name"] == name
                and record["date"] == date_str
            ):
                return False

        with open(
            ATTENDANCE_FILE,
            mode="a",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                name,
                date_str,
                time_str
            ])

    return True


def get_today_attendance():
    """
    Return today's attendance records.
    """

    today = datetime.now().strftime("%Y-%m-%d")

    records = get_attendance_records()

    return [
        record
        for record in records
        if record["date"] == today
    ]


 
# FACE RECOGNITION
 

def recognize_face(image):
    """
    Recognize faces in an RGB image.

    Returns:
        {
            "name": str,
            "recognized": bool,
            "location": [top, right, bottom, left]
        }
    """

    global known_face_encodings
    global known_face_names

    face_locations = face_recognition.face_locations(
        image,
        model="hog"
    )

    face_encodings = face_recognition.face_encodings(
        image,
        face_locations
    )

    results = []

    for face_encoding, face_location in zip(
        face_encodings,
        face_locations
    ):

        name = "Unknown Student"
        recognized = False

        if len(known_face_encodings) > 0:

            face_distances = face_recognition.face_distance(
                known_face_encodings,
                face_encoding
            )

            best_match_index = np.argmin(
                face_distances
            )

            distance = face_distances[
                best_match_index
            ]

            # 0.50 is relatively strict
            if distance <= 0.50:

                name = known_face_names[
                    best_match_index
                ]

                recognized = True

        results.append({
            "name": name,
            "recognized": recognized,
            "location": list(face_location)
        })

    return results


 
# MAIN PORTAL
 

@app.route("/")
def portal():

    return render_template(
        "portal.html"
    )


 
# USER SCANNER
 

@app.route("/user")
def user_scanner():

    return render_template(
        "index.html"
    )


 
# FACE RECOGNITION API
 

@app.route("/recognize", methods=["POST"])
def recognize():

    try:

        data = request.get_json(
            silent=True
        )

        if not data or "image" not in data:

            return jsonify({
                "success": False,
                "error": "No image received"
            }), 400

        image_data = data["image"]

        # Remove:
        # data:image/jpeg;base64,
        if "," in image_data:

            image_data = image_data.split(
                ",",
                1
            )[1]

        # Decode base64
        image_bytes = base64.b64decode(
            image_data
        )

        # Open image using Pillow
        pil_image = Image.open(
            BytesIO(image_bytes)
        ).convert("RGB")

        # Convert to numpy array
        rgb_image = np.array(
            pil_image
        )

        # Recognition
        results = recognize_face(
            rgb_image
        )

        response_results = []

        for result in results:

            name = result["name"]
            recognized = result["recognized"]

            attendance_marked = False

            if recognized:

                attendance_marked = mark_attendance(
                    name
                )

            response_results.append({

                "name": name,

                "recognized": recognized,

                "attendance_marked":
                    attendance_marked,

                "location":
                    result["location"]
            })

        # If no faces
        if len(response_results) == 0:

            return jsonify({

                "success": True,

                "face_detected": False,

                "results": []

            })

        return jsonify({

            "success": True,

            "face_detected": True,

            "results": response_results

        })

    except Exception as e:

        print(
            f"[RECOGNITION ERROR] {e}"
        )

        return jsonify({

            "success": False,

            "error":
                "Could not process image"

        }), 500


 
# STATISTICS API
 

@app.route("/api/stats")
def api_stats():

    try:

        # Total registered students
        total_students = len(
            set(known_face_names)
        )

        # Today's attendance
        today_records = get_today_attendance()

        present_today = len(
            today_records
        )

        # Last scanned student
        all_records = get_attendance_records()

        if all_records:

            last_record = all_records[-1]

            last_scanned = last_record["name"]

            last_scanned_time = (
                last_record["time"]
            )

        else:

            last_scanned = ""

            last_scanned_time = "--:--:--"

        return jsonify({

            "total_students":
                total_students,

            "present_today":
                present_today,

            "last_scanned":
                last_scanned,

            "last_scanned_time":
                last_scanned_time

        })

    except Exception as e:

        print(
            f"[STATS ERROR] {e}"
        )

        return jsonify({

            "total_students": 0,

            "present_today": 0,

            "last_scanned": "",

            "last_scanned_time":
                "--:--:--"

        })


 
# ADMIN LOGIN
 

@app.route(
    "/admin/login",
    methods=["GET", "POST"]
)
def admin_login():

    if session.get(
        "admin_authenticated"
    ):

        return redirect(
            url_for("admin_dashboard")
        )

    if request.method == "POST":

        password = request.form.get(
            "password",
            ""
        )

        if password == ADMIN_PASSWORD:

            session[
                "admin_authenticated"
            ] = True

            return redirect(
                url_for("admin_dashboard")
            )

        flash(
            "Incorrect administrator password.",
            "error"
        )

    return render_template(
        "admin_login.html"
    )


 
# ADMIN DASHBOARD
 

@app.route("/admin")
@admin_required
def admin_dashboard():

    today_records = get_today_attendance()

    return render_template(

        "admin.html",

        total_students=len(
            set(known_face_names)
        ),

        present_today=len(
            today_records
        )

    )


 
# STUDENT REGISTRATION
 

@app.route(
    "/register",
    methods=["GET", "POST"]
)
@admin_required
def register_student():

    if request.method == "GET":

        return render_template(
            "register.html"
        )

    try:

        data = request.get_json(
            silent=True
        )

        if not data:

            return jsonify({

                "success": False,

                "error":
                    "Invalid request"

            }), 400

        name = data.get(
            "name",
            ""
        ).strip()

        image_data = data.get(
            "image",
            ""
        )

        # Validate name
        if not name:

            return jsonify({

                "success": False,

                "error":
                    "Please enter the student's name."

            }), 400

        if not image_data:

            return jsonify({

                "success": False,

                "error":
                    "No photograph received."

            }), 400

        # Create safe filename
        safe_name = re.sub(
            r"[^A-Za-z0-9]+",
            "_",
            name
        ).strip("_")

        if not safe_name:

            return jsonify({

                "success": False,

                "error":
                    "Invalid student name."

            }), 400

        filename = (
            f"{safe_name}.jpg"
        )

        filepath = os.path.join(
            DATASET_DIR,
            filename
        )

        # Decode image
        if "," in image_data:

            image_data = image_data.split(
                ",",
                1
            )[1]

        image_bytes = base64.b64decode(
            image_data
        )

        pil_image = Image.open(
            BytesIO(image_bytes)
        ).convert("RGB")

        rgb_image = np.array(
            pil_image
        )

        # Make sure exactly one face exists
        face_locations = (
            face_recognition.face_locations(
                rgb_image
            )
        )

        if len(face_locations) == 0:

            return jsonify({

                "success": False,

                "error":
                    "No face detected. Please look directly at the camera."

            }), 400

        if len(face_locations) > 1:

            return jsonify({

                "success": False,

                "error":
                    "Multiple faces detected. Please keep only one person in the frame."

            }), 400

        # Save JPEG
        pil_image.save(
            filepath,
            "JPEG",
            quality=95
        )

        # Immediately reload face database
        reload_face_database()

        return jsonify({

            "success": True,

            "message":
                f"{name} registered successfully.",

            "name": name

        })

    except Exception as e:

        print(
            f"[REGISTRATION ERROR] {e}"
        )

        return jsonify({

            "success": False,

            "error":
                "Registration failed."

        }), 500


 
# ATTENDANCE LOG
 

@app.route("/attendance")
@admin_required
def view_attendance():

    records = get_attendance_records()

    # Newest first
    records.reverse()

    return render_template(

        "attendance.html",

        records=records

    )


 
# RELOAD DATASET
 

@app.route("/reload_dataset")
@admin_required
def reload_dataset():

    reload_face_database()

    flash(
        "Face database reloaded successfully.",
        "success"
    )

    return redirect(
        url_for("admin_dashboard")
    )


 
# ADMIN LOGOUT
 

@app.route("/admin/logout")
def admin_logout():

    session.pop(
        "admin_authenticated",
        None
    )

    return redirect(
        url_for("portal")
    )


# DEVELOPMENT SERVER


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
