
import pandas as pd
import numpy as np
import pickle

from flask import Flask, render_template, request

app = Flask(__name__)
app.config["SECRET_KEY"] = "jobshield-secret-key"

# Load trained machine learning files
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

# Dashboard statistics (temporary, reset when server restarts)
stats = {
    "total": 0,
    "genuine": 0,
    "fraudulent": 0
}


@app.route("/")
def Home():
    return render_template(
        "index.html",
        result=None,
        stats=stats
    )


@app.route("/submit", methods=["POST"])
def submit():

    # Collect all job details
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

    # Convert job details into numerical features
    input_features = vectorizer.transform(input_data)

    # Predict job classification
    prediction = model.predict(input_features)[0]

    # Get probability for each model class, if supported
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

    # Determine result
    if prediction == 1:
        status = "FRAUDULENT JOB"
        stats["fraudulent"] += 1
    else:
        status = "REAL JOB"
        stats["genuine"] += 1

    stats["total"] += 1

    # Prepare dashboard result
    result = {
        "status": status,
        "safety_score": safety_score,
        "fraud_score": fraud_score
    }

    return render_template(
        "index.html",
        result=result,
        stats=stats
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
