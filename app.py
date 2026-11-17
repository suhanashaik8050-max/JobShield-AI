
import os
import re
import pickle
import io

import pandas as pd
import numpy as np
import fitz
import pytesseract

from PIL import Image
from flask import Flask, render_template, request


app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get(
    "SECRET_KEY", "jobshield-secret-key"
)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg"}

# Load trained machine learning files
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)


# Dashboard statistics
stats = {
    "total": 0,
    "genuine": 0,
    "fraudulent": 0
}


# Check uploaded file extension
def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


# Extract text from PDF and images
def extract_text(file, filename):

    extension = filename.rsplit(".", 1)[1].lower()
    file_bytes = file.read()

    if extension == "pdf":

        document = fitz.open(
            stream=file_bytes,
            filetype="pdf"
        )

        extracted_text = ""

        for page in document:
            page_text = page.get_text()

            # OCR for scanned PDF pages
            if not page_text.strip():
                pix = page.get_pixmap(
                    matrix=fitz.Matrix(2, 2)
                )

                image = Image.open(
                    io.BytesIO(pix.tobytes("png"))
                )

                page_text = pytesseract.image_to_string(image)

            extracted_text += page_text + "\n"

        document.close()
        return extracted_text

    else:

        image = Image.open(io.BytesIO(file_bytes))
        image = image.convert("RGB")

        return pytesseract.image_to_string(image)


# Analyze suspicious offer letter indicators
def analyze_offer_letter(text):

    text_lower = text.lower()

    warnings = []
    risk_score = 0

    checks = [
        (
            r"registration fee|processing fee|joining fee|"
            r"security deposit|training fee|pay.*before joining",
            "Requests for registration, processing or joining fees.",
            30
        ),
        (
            r"personal bank account|send money|transfer money|"
            r"pay now|payment required",
            "Suspicious payment or money-transfer instructions.",
            25
        ),
        (
            r"gmail\.com|yahoo\.com|hotmail\.com|outlook\.com",
            "Personal email address detected. Verify the sender independently.",
            15
        ),
        (
            r"limited time|act immediately|urgent payment|"
            r"offer expires today",
            "Urgent or high-pressure language detected.",
            15
        ),
        (
            r"guaranteed job|no interview required|"
            r"instant selection",
            "Unusual job selection promises detected.",
            15
        ),
        (
            r"otp|password|banking credentials|"
            r"card details",
            "Sensitive personal or banking information may be requested.",
            30
        )
    ]

    for pattern, warning, points in checks:

        if re.search(pattern, text_lower):

            warnings.append(warning)
            risk_score += points

    # Check important offer letter details
    missing_details = []

    if not re.search(
        r"company|organization|pvt\.?\s*ltd|limited|llp",
        text_lower
    ):
        missing_details.append("Company information")

    if not re.search(
        r"salary|stipend|compensation|remuneration",
        text_lower
    ):
        missing_details.append("Salary or stipend details")

    if not re.search(
        r"date|joining|commencement",
        text_lower
    ):
        missing_details.append("Joining date")

    if missing_details:
        warnings.append(
            "Missing or unclear details: "
            + ", ".join(missing_details)
        )
        risk_score += min(len(missing_details) * 10, 30)

    risk_score = min(risk_score, 100)

    if risk_score >= 60:
        status = "HIGH RISK"
    elif risk_score >= 30:
        status = "MEDIUM RISK"
    else:
        status = "LOW RISK"

    if not warnings:
        warnings.append(
            "No major suspicious indicators were detected "
            "by the current rule-based analysis."
        )

    return {
        "risk_score": risk_score,
        "status": status,
        "warnings": warnings,
        "missing_details": missing_details
    }


# Home page
@app.route("/")
def Home():

    return render_template(
        "index.html",
        result=None,
        stats=stats,
        offer_result=None
    )


# Existing fake job detection feature
@app.route("/submit", methods=["POST"])
def submit():

    input_data = [
        request.form.get("title", "") + " " +
        request.form.get("location", "") + " " +
        request.form.get("department", "") + " " +
        request.form.get("profile", "") + " " +
        request.form.get("req", "") + " " +
        request.form.get("ben", "") + " " +
        request.form.get("emptype", "") + " " +
        request.form.get("exp", "") + " " +
        request.form.get("edu", "") + " " +
        request.form.get("indu", "") + " " +
        request.form.get("func", "") + " " +
        request.form.get("des", "")
    ]

    input_features = vectorizer.transform(input_data)

    prediction = model.predict(input_features)[0]

    safety_score = None
    fraud_score = None

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(input_features)[0]
        classes = list(model.classes_)

        if 0 in classes:
            safety_score = round(
                float(probabilities[classes.index(0)]) * 100
            )

        if 1 in classes:
            fraud_score = round(
                float(probabilities[classes.index(1)]) * 100
            )

    if prediction == 1:

        status = "FRAUDULENT JOB"
        stats["fraudulent"] += 1

    else:

        status = "REAL JOB"
        stats["genuine"] += 1

    stats["total"] += 1

    result = {
        "status": status,
        "safety_score": safety_score,
        "fraud_score": fraud_score
    }

    return render_template(
        "index.html",
        result=result,
        stats=stats,
        offer_result=None
    )


# NEW FEATURE: Offer Letter Verification
@app.route("/verify-offer-letter", methods=["POST"])
def verify_offer_letter():

    uploaded_file = request.files.get("offer_letter")

    if not uploaded_file or uploaded_file.filename == "":
        return render_template(
            "index.html",
            result=None,
            stats=stats,
            offer_result={
                "error": "Please select an offer letter to upload."
            }
        )

    if not allowed_file(uploaded_file.filename):
        return render_template(
            "index.html",
            result=None,
            stats=stats,
            offer_result={
                "error": "Only PDF, JPG, JPEG and PNG files are allowed."
            }
        )

    try:

        extracted_text = extract_text(
            uploaded_file,
            uploaded_file.filename
        )

        if len(extracted_text.strip()) < 20:

            offer_result = {
                "error": "Could not extract enough text. Please upload a clearer document."
            }

        else:

            analysis = analyze_offer_letter(extracted_text)

            offer_result = {
                "filename": uploaded_file.filename,
                "risk_score": analysis["risk_score"],
                "status": analysis["status"],
                "warnings": analysis["warnings"],
                "missing_details": analysis["missing_details"]
            }

    except Exception as error:

        print("Offer letter processing error:", error)

        offer_result = {
            "error": "Unable to process this document. Please try another file."
        }

    return render_template(
        "index.html",
        result=None,
        stats=stats,
        offer_result=offer_result
    )


# File size error
@app.errorhandler(413)
def file_too_large(error):

    return "File too large. Maximum upload size is 10 MB.", 413


if __name__ == "__main__":

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )
