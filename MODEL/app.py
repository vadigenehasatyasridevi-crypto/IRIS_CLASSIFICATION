import os
from pathlib import Path

import joblib
import numpy as np

from flask import Flask, render_template, request, jsonify


app = Flask(__name__)


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

MODEL_PATH = Path(__file__).resolve().parent / "iris_model.pkl"

model_data = joblib.load(MODEL_PATH)

model = model_data["model"]
target_names = model_data["target_names"]


# --------------------------------------------------
# Home page
# --------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# --------------------------------------------------
# Prediction API
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Get JSON data
        data = request.get_json()

        # Extract features
        sepal_length = float(data["sepal_length"])
        sepal_width = float(data["sepal_width"])
        petal_length = float(data["petal_length"])
        petal_width = float(data["petal_width"])

        # Create input array
        input_data = np.array([
            [
                sepal_length,
                sepal_width,
                petal_length,
                petal_width
            ]
        ])

        # Prediction
        prediction = model.predict(input_data)[0]

        # Convert prediction to class name
        predicted_class = target_names[prediction]

        return jsonify({
            "prediction": f"Iris-{predicted_class}"
        })

    except (KeyError, TypeError, ValueError) as error:

        return jsonify({
            "error": f"Invalid input: {error}"
        }), 400


# --------------------------------------------------
# Run application
# --------------------------------------------------

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port
    )