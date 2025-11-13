import joblib
import pandas as pd
import os
import logging
import webbrowser
import threading
from flask import Flask, request, jsonify, render_template

# --------------------------------------
# CONFIG
# --------------------------------------
BASE_DIR = r"C:\Users\aghin\OneDrive\Documents\IBM FINAL\IBM PROJECT"
MODEL_FILE = r"C:\Users\aghin\OneDrive\Documents\IBM FINAL\IBM PROJECT\salary_prediction_model_enhanced.pkl"

# Full model path
model_path = os.path.join(BASE_DIR, MODEL_FILE)

# --------------------------------------
# FLASK SETUP
# --------------------------------------
app = Flask(__name__)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# --------------------------------------
# LOAD MODEL
# --------------------------------------
try:
    model = joblib.load(model_path)
    logging.info(f"Model loaded successfully from {model_path}")
except Exception as e:
    model = None
    logging.error(f"ERROR: Unable to load model. Details: {e}")


# --------------------------------------
# ROUTES
# --------------------------------------
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json(force=True)

        job_title = data.get("jobTitle")
        education = data.get("education")
        gender = data.get("gender")
        age = int(data.get("age"))
        experience = int(data.get("experience"))

        # Validation: Experience cannot exceed Age - 18
        max_exp = age - 18
        if experience > max_exp:
            return jsonify({
                "error": f"Experience cannot exceed {max_exp} years for age {age}"
            }), 400

        # Create DataFrame
        df = pd.DataFrame([{
            "Job Title": job_title,
            "Education Level": education,
            "Gender": gender,
            "Age": age,
            "Years of Experience": experience
        }])

        # Predict
        prediction = float(model.predict(df)[0])

        return jsonify({"predicted_salary": round(prediction, 2)})

    except Exception as e:
        logging.error(f"Prediction Error: {e}")
        return jsonify({"error": "Internal Server Error"}), 500


# --------------------------------------
# RUN SERVER
# --------------------------------------
if __name__ == '__main__':
    url = "http://127.0.0.1:5001"
    threading.Timer(1.25, lambda: webbrowser.open(url)).start()
    app.run(host="0.0.0.0", port=5001, debug=True)
